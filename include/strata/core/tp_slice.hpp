// include/strata/core/tp_slice.hpp - the tensor split: what HALF of every layer is, as a geometry, as weights and
// as session state.
//
// THE KERNELS DO NOT CHANGE.  Every kernel of the verify window takes its head counts, channel counts and ff width
// from `ModelGeometry` at run time, so a half runs the very same window with the half geometry below and weights
// cut to match.  The two halves' partial sums are added after the mixer's out-projection and after the MoE
// (`tp_exchange.hpp`); everything between two exchanges - hyper-connections, router, indexer - is replicated and,
// because the exchange gives both halves a bitwise identical sum, computes the same thing on both.
//
// WHICH HEADS A HALF TAKES.
//   * GDN pairs value head v with key head `v % h_k` (MODULO, verify_kernels.cu `qh = head % h_k`), so half c
//     takes key heads [c*h_k/2, (c+1)*h_k/2) and the value heads {i*h_k + c*h_k/2 + j}: three groups of eight on
//     this model.  Local value head i*(h_k/2) + j then pairs with local key head j, exactly as the kernel expects.
//   * QSA maps query head h to kv head h / (n_head/n_head_kv) - contiguous - so half c takes query heads
//     [c*n_head/2, ...) and kv head c.  attn_q interleaves [q_h | gate_h] per head, so its rows are contiguous too.
//   * The shared expert takes ff rows [c*n_ff/2, ...).
// Output-row cuts copy whole rows; input cuts (ssm_out, attn_output, ffn_down_shexp) keep a block range of every
// row, so the quant type never changes - a cut that is not a whole number of blocks is refused.
#pragma once

#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

struct TpRange { int64_t begin = 0, end = 0; };

/// Half of `g`: key/value heads, conv channels, value dim, query/kv heads and ff width halved; the rest kept.
/// False (with `err`) when a count does not split evenly or the split would break a kernel's assumption.
bool tp_half_geometry(const ModelGeometry& g, ModelGeometry& half, std::string& err);

/// Half c's rows of a GDN layer's q|k|v channels (attn_qkv rows, ssm_conv1d rows, the conv state).
std::vector<TpRange> tp_gdn_channels(const ModelGeometry& g, int half);
/// Half c's GDN value heads (ranges of heads).
std::vector<TpRange> tp_gdn_vheads(const ModelGeometry& g, int half);

/// The weight table of one half: a copy of the full table whose cut tensors point at this object's own device
/// copies (their native GGUF rows cut from the shards; the small canonical GDN tensors gathered from the full
/// arena).  Everything replicated aliases the full table, which must outlive this.
class TpWeights {
public:
    TpWeights() = default;
    ~TpWeights();
    TpWeights(const TpWeights&) = delete;
    TpWeights& operator=(const TpWeights&) = delete;

    bool build(const WeightTable& full, const std::vector<std::string>& shards, const ModelGeometry& g, int half,
               std::string& err);
    const WeightTable& table() const { return table_; }
    /// This half's own q8_1 scratch (the size the full table's native projections need), for the PLE key.
    void* q8_1() const { return q8_1_; }
    uint64_t bytes() const { return bytes_; }
    size_t tensors() const { return cut_; }

private:
    WeightTable table_;
    std::vector<void*> allocs_;
    void* q8_1_ = nullptr;
    uint64_t bytes_ = 0;
    size_t cut_ = 0;
};

/// The VRAM expert arena of one half: slot s holds half c of the expert the full cache's slot s holds - its gate and
/// up rows [c*n_ff/2, (c+1)*n_ff/2) and the same half of every down row's K - which is exactly the native blob of an
/// expert with ff width n_ff/2 (`native_expert_layout(gu, d, n_embd, n_ff/2)`), half the full blob's bytes.  Slots
/// are sized by the layer they hold (a slot never changes layer), 256-byte aligned like `ExpertCache::open_sized`.
class TpExpertHalves {
public:
    TpExpertHalves() = default;
    ~TpExpertHalves();
    TpExpertHalves(const TpExpertHalves&) = delete;
    TpExpertHalves& operator=(const TpExpertHalves&) = delete;

    /// `slot_layer[s]`: the layer slot s holds, or -1 for a slot never filled (no bytes).  On the current device.
    bool open(const std::vector<int32_t>& slot_layer, int half, std::string& err);
    /// Half c of the full native blob `full` (device memory, mapped or registered host memory, or pageable host
    /// memory - `cudaMemcpyDefault`) into `slot`: gate and up as one piece each, down as a 2D copy of every row's
    /// half.  Asynchronous on `stream`.
    bool fill(int32_t slot, const uint8_t* full, void* stream, std::string& err);
    /// The same from a HOST blob (the arena): the whole blob in one contiguous copy into a small device ring, then
    /// the gather on the device.  A 2D copy straight from host memory went row by row: 5.77 ms per expert for both
    /// halves against 0.27 for the whole contiguous blob [measured 30.09].  The ring is reused in stream order.
    bool fill_host(int32_t slot, const uint8_t* full_host, void* stream, std::string& err);
    /// Reads `slot` back and compares it byte for byte with half c gathered on the host from `full_host`.
    bool verify_slot(int32_t slot, const uint8_t* full_host, std::string& err) const;

    const uint8_t* base() const { return base_; }
    const uint64_t* slot_offsets() const { return off_.data(); }
    uint64_t bytes() const { return off_.empty() ? 0 : off_.back(); }
    int32_t layer_of(int32_t slot) const { return slot >= 0 && (size_t) slot < layer_.size() ? layer_[(size_t) slot] : -1; }

private:
    uint8_t* base_ = nullptr;
    std::vector<uint64_t> off_;
    std::vector<int32_t> layer_;
    int half_ = 0;
    static constexpr int kRing = 4;
    uint8_t* ring_ = nullptr;
    size_t ring_blob_ = 0;
    int ring_next_ = 0;
};

/// Which layer each of the full cache's `n_slots` slots holds, from the residency table (-1: none).
std::vector<int32_t> tp_slot_layers(const std::vector<int32_t>& host_res, int64_t n_layers, int64_t n_expert,
                                    int64_t n_slots);

/// Copy the full session's state into half c's session (session_init'ed with the half geometry): the GDN
/// recurrence and conv history of this half's heads/channels, the QSA KV of its kv head, the indexer state, the
/// PLE history and window, the residual.  The PLE run is wired to the half's own history and `ple_scratch` /
/// `ple_q8_1` (the PLE block's workspaces - two halves running at once cannot share them).  Synchronous.
bool tp_split_state(const ModelGeometry& g, const SessionState& full, const ModelGeometry& gh, SessionState& half,
                    int c, void* ple_scratch, void* ple_q8_1, std::string& err);

}  // namespace strata::core
