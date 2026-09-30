// include/strata/core/tp_pair.hpp - the two halves of a tensor split as ONE object, for both the speculative loop
// and the server (tp_slice.hpp says what a half is).
//
// It owns each half's cut weights, its routed experts' halves, its session (half geometry) and its verifier, plus
// the refill stream of GPU 1.  Half 0 stands in for the whole-layer verifier: `vt[0].run` / `vt[0].commit` drive
// half 1 as well (Verifier::tp_pair).
//
// ORDER: open -> (halves_from_cache | nothing) -> build -> (own_cache) -> init_verifiers.  `own_cache` sizes the
// halves' arena by what is free on the tighter card, so it comes after everything else of the halves is allocated.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/tp_slice.hpp"
#include "strata/core/verify.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

class TpPair {
public:
    TpPair() = default;
    ~TpPair();
    TpPair(const TpPair&) = delete;
    TpPair& operator=(const TpPair&) = delete;

    /// `pair`: half c on GPU c (P2P); else both halves on the current GPU.  `own`: two GPUs with the halves' own
    /// expert arena filled from the host (no full cache); else the halves of the full cache's slots.
    bool open(const ModelGeometry& g, bool pair, bool own, std::string& err);
    /// Not own: half c of every filled slot of the full cache (GPU 0 in place, GPU 1 through its ring).
    bool halves_from_cache(const ModelGeometry& g, const std::vector<int32_t>& host_res, ExpertCache& xcache,
                           std::string& err);
    /// Each half's cut weights, replicas (GPU 1), session (half geometry) and PLE workspaces; the sessions split from
    /// `ss` (every position).
    bool build(const WeightTable& wt, const std::vector<std::string>& shards, const ModelGeometry& g,
               const SessionState& ss, int64_t max_context, int64_t k, std::string& err);
    /// Own: as many halves as the tighter card holds after `reserve_mib`, ranked by the profile, filled from `src`
    /// (the arena's pages of what is now in VRAM are released).  `res` is then the table the pool reads.
    bool own_cache(const ModelGeometry& g, ExpertSource& src, const std::string& profile, int64_t reserve_mib,
                   std::string& err);
    /// Both verifiers (half 0 with the head), then paired.  `vh`: the full cache's hits (the mark and the device
    /// table); each half gets its own arena.
    bool init_verifiers(const ModelGeometry& g, const VerifyHits& vh, const std::vector<int32_t>& host_res,
                        const NativeHead* head, int spec, std::string& err);

    /// The table the pool and the swaps use: `res` (own) or the full cache's.
    std::vector<int32_t>& table(std::vector<int32_t>& host_res) { return own ? res : host_res; }

    /// The adaptive tier's swaps: both halves of every (slot, full blob in host memory) - half 0 on `stream0`,
    /// half 1 on GPU 1's own stream (two GPUs) or `stream0`.  Timed; `fills_done` says when both have landed.
    bool fill(const std::vector<std::pair<int32_t, const uint8_t*>>& fills, void* stream0, std::string& err);
    bool fills_done(bool wait);
    /// Every half of `fills` byte for byte against its blob (STRATA_TP_VERIFY_SWAPS).
    bool verify(const std::vector<std::pair<int32_t, const uint8_t*>>& fills, std::string& err);

    bool pair = false, own = false;
    int dev[2] = {0, 0};
    ModelGeometry gh;
    TpWeights tw[2];
    TpExpertHalves th[2];
    SessionState hs[2];
    Verifier vt[2];
    std::vector<int32_t> res;   ///< own: (layer, expert) -> slot of the halves' arena

    int64_t swapped = 0, batches = 0, verified = 0;
    double ms_half0 = 0, ms_half1 = 0;   ///< the swaps' copies on each GPU

private:
    std::vector<void*> mem_;   ///< sessions, PLE workspaces, GPU 1's residency table (freed with this)
    std::vector<int> mem_dev_;
    void* ple_scratch_[2] = {nullptr, nullptr};
    void* stream1_ = nullptr;
    void* ev_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};   ///< [gpu][begin/end] of the batch in flight
    int64_t in_flight_ = 0;
};

}  // namespace strata::core
