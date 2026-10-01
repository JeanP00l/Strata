#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata::core {
class WeightTable;

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Owns one Q8_1 scratch vector shared by all
// these projections, so use one ordered session stream and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    /// `host`: every projection except the PLE key in pinned, mapped host memory (`WeightRef::native_host`) - a
    /// tensor split's halves hold their own cuts, and the whole-layer tensors serve only the prompt path, which
    /// copies each to the card before its GEMM.  The PLE key stays in VRAM: the halves' PLE block reads it.
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false, bool host = false);
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key,
                             std::set<std::string>& out, std::string& err);
    /// Layer split: load only blocks [lb, le) (every other `blk.N.` projection belongs to another GPU's stage).
    /// Process-wide, read by the next `load`; (-1, -1) = all layers.
    static void set_layer_range(int lb, int le);
    /// Frees every projection not in `keep` (tensor split on two GPUs: the halves hold their own cuts, and the
    /// whole-layer tensors are needed only by the prompt path).  The table's refs to them dangle afterwards.
    uint64_t release_except(const std::set<const void*>& keep);
    /// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized (iq_pack --compat-bf16 of a GGUF key
    /// the native kernel also reads, e.g. OrcaRouter's IQ3_XXS) serves the PLE from that row, so it is taken out
    /// of `skip` and `load` does not upload the GGUF key over it.  A quantized row leaves `skip` unchanged.
    static bool keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err);
    uint64_t weight_bytes() const { return bytes_; }
    /// Of those, in pinned host memory (`load(..., host = true)`), and the largest one (the prompt path's staging).
    uint64_t host_bytes() const { return host_bytes_; }
    uint64_t largest_host() const { return largest_host_; }
    size_t tensor_count() const { return weights_.size(); }

private:
    std::vector<void*> weights_;
    std::vector<uint64_t> sizes_;
    std::vector<bool> host_;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0, host_bytes_ = 0, largest_host_ = 0;
};
} // namespace strata::core
