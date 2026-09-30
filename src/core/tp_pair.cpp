// src/core/tp_pair.cpp - see include/strata/core/tp_pair.hpp.
#include "strata/core/tp_pair.hpp"

#include "strata/core/layer.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/tp_exchange.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>

namespace strata::core {

TpPair::~TpPair() {
    for (int c = 0; c < 2; ++c)
        for (void*& e : ev_[c])
            if (e != nullptr) { const OnDevice on(dev[c]); cudaEventDestroy((cudaEvent_t) e); }
    if (stream1_ != nullptr) { const OnDevice on(dev[1]); cudaStreamDestroy((cudaStream_t) stream1_); }
    for (size_t i = 0; i < mem_.size(); ++i) { const OnDevice on(mem_dev_[i]); cudaFree(mem_[i]); }
}

bool TpPair::open(const ModelGeometry& g, bool pair_gpus, bool own_cache_, std::string& err) {
    pair = pair_gpus;
    own = pair_gpus && own_cache_;
    dev[0] = 0;
    dev[1] = pair ? 1 : 0;
    if (!tp_half_geometry(g, gh, err)) return false;
    if (pair) {
        int n_dev = 0;
        cudaGetDeviceCount(&n_dev);
        if (n_dev < 2 || !strata::kernels::tp_enable_peer(0, 1)) {
            err = "--tensor-split pair needs two GPUs with peer access (" + std::to_string(n_dev) + " visible)";
            return false;
        }
    }
    for (int c = 0; c < 2; ++c) {
        const OnDevice on(dev[c]);
        if (cudaEventCreate((cudaEvent_t*) &ev_[c][0]) != cudaSuccess || cudaEventCreate((cudaEvent_t*) &ev_[c][1]) != cudaSuccess) {
            err = "tensor split: cannot create the refill events";
            return false;
        }
    }
    if (pair) {
        const OnDevice on(dev[1]);
        if (cudaStreamCreateWithFlags((cudaStream_t*) &stream1_, cudaStreamNonBlocking) != cudaSuccess) {
            err = "tensor split: cannot create GPU 1's refill stream";
            return false;
        }
    }
    return true;
}

bool TpPair::halves_from_cache(const ModelGeometry& g, const std::vector<int32_t>& host_res, ExpertCache& xcache,
                               std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<int32_t> sl = tp_slot_layers(host_res, g.n_layers, g.n_expert, xcache.slots());
    for (int c = 0; c < 2; ++c) {
        const OnDevice on(dev[c]);
        if (!th[c].open(sl, c, err)) return false;
        for (int32_t s = 0; s < (int32_t) sl.size(); ++s)
            if (sl[(size_t) s] >= 0 &&
                !(dev[c] == 0 ? th[c].fill(s, xcache.device_slot(s), nullptr, err)
                              : th[c].fill_staged(s, xcache.device_slot(s), nullptr, err)))
                return false;
        if (cudaDeviceSynchronize() != cudaSuccess) {
            err = "tensor split: the expert halves' copies failed (GPU " + std::to_string(dev[c]) + ")";
            return false;
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    int32_t s0 = 0;
    while (s0 < (int32_t) sl.size() && sl[(size_t) s0] < 0) ++s0;
    if (s0 < (int32_t) sl.size()) {   // the first and the last filled slot, both halves, byte for byte
        for (int32_t s : {s0, (int32_t) sl.size() - 1}) {
            if (sl[(size_t) s] < 0) continue;
            std::vector<uint8_t> full((size_t) strata::kernels::cpu::expert_layout().blob_bytes(sl[(size_t) s]));
            if (cudaMemcpy(full.data(), xcache.device_slot(s), full.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
                err = "tensor split: read-back failed";
                return false;
            }
            if (!th[0].verify_slot(s, full.data(), err) || !th[1].verify_slot(s, full.data(), err)) return false;
        }
    }
    std::fprintf(stderr, "strata generate: tensor split experts: %zu slots halved, 2 x %.2f GiB, %.0f ms; "
                         "first and last slot verified\n",
                 sl.size(), (double) th[0].bytes() / 1073741824.0, ms);
    return true;
}

bool TpPair::build(const WeightTable& wt, const std::vector<std::string>& shards, const ModelGeometry& g,
                   const SessionState& ss, int64_t max_context, int64_t k, std::string& err) {
    for (int c = 0; c < 2; ++c) {
        const OnDevice on(dev[c]);
        void* sb = nullptr;
        void* ps = nullptr;
        if (!tw[c].build(wt, shards, g, c, err) || (dev[c] != 0 && !tw[c].localize(err))) {
            err = "tensor split half " + std::to_string(c) + ": " + err;
            return false;
        }
        if (cudaMalloc(&sb, session_bytes(gh, max_context, k)) != cudaSuccess) {
            err = "tensor split half " + std::to_string(c) + ": out of device memory (raise --vram-reserve-mib)";
            return false;
        }
        mem_.push_back(sb);
        mem_dev_.push_back(dev[c]);
        if (cudaMalloc(&ps, ple_run_scratch_bytes()) != cudaSuccess) {
            err = "tensor split half " + std::to_string(c) + ": out of device memory (raise --vram-reserve-mib)";
            return false;
        }
        mem_.push_back(ps);
        mem_dev_.push_back(dev[c]);
        ple_scratch_[c] = ps;
        if (session_init(gh, max_context, k, sb, hs[c]) == 0 ||
            !tp_split_state(g, ss, gh, hs[c], c, ps, tw[c].q8_1(), err)) {
            err = "tensor split half " + std::to_string(c) + ": " + err;
            return false;
        }
        if (dev[c] != 0 && hs[c].ple.ready()) {   // the PLE block's weights: this GPU's replicas
            auto& w = hs[c].ple.w;
            w.key_codes = (const uint8_t*) tw[c].local(w.key_codes);
            w.key_scales = (const float*) tw[c].local(w.key_scales);
            w.value_bf16 = (const uint16_t*) tw[c].local(w.value_bf16);
            w.norm_key = (const float*) tw[c].local(w.norm_key);
            w.norm_query = (const float*) tw[c].local(w.norm_query);
            w.norm_conv = (const float*) tw[c].local(w.norm_conv);
            w.conv1d_f16 = (const uint16_t*) tw[c].local(w.conv1d_f16);
            w.key_native_data = tw[c].local(w.key_native_data);
            w.key_bf16 = (const uint16_t*) tw[c].local(w.key_bf16);
            for (const void* q : {(const void*) w.key_codes, (const void*) w.key_scales, (const void*) w.value_bf16,
                                  (const void*) w.norm_key, (const void*) w.norm_query, (const void*) w.norm_conv,
                                  (const void*) w.conv1d_f16, w.key_native_data, (const void*) w.key_bf16})
                if (q != nullptr && strata::kernels::tp_pointer_device(q) >= 0 &&
                    strata::kernels::tp_pointer_device(q) != dev[c]) {
                    err = "tensor split half " + std::to_string(c) + ": a PLE weight is not in the table and stays on "
                          "another GPU";
                    return false;
                }
        }
    }
    return true;
}

bool TpPair::own_cache(const ModelGeometry& g, ExpertSource& src, const std::string& profile, int64_t reserve_mib,
                       std::string& err) {
    std::vector<std::pair<int32_t, int32_t>> ranked;
    int64_t pslots = 0;
    if (profile.empty()) { err = "tensor split pair: needs --expert-profile"; return false; }
    if (!read_expert_profile(profile, g.n_layers, g.n_expert, ranked, pslots, err)) {
        err = "tensor split pair: " + err;
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    int64_t budget = INT64_MAX;
    for (int c = 0; c < 2; ++c) {
        const OnDevice on(dev[c]);
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        budget = std::min<int64_t>(budget, (int64_t) fb - (reserve_mib << 20));
        std::fprintf(stderr, "strata generate: tensor split pair: GPU %d has %.2f GiB free\n", dev[c], (double) fb / 1073741824.0);
    }
    std::vector<int32_t> sl;
    int64_t used = 0;
    for (const auto& pr : ranked) {
        const int64_t b = ((int64_t) lay.blob_bytes(pr.first) / 2 + 255) / 256 * 256;
        if (used + b > budget) break;
        used += b;
        sl.push_back(pr.first);
    }
    if (sl.empty()) { err = "tensor split pair: no room for experts"; return false; }
    res.assign((size_t) (g.n_layers * g.n_expert), kNotResident);
    for (size_t i = 0; i < sl.size(); ++i)
        res[(size_t) ranked[i].first * (size_t) g.n_expert + (size_t) ranked[i].second] = (int32_t) i;
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < 2; ++c) {
        const OnDevice on(dev[c]);
        if (!th[c].open(sl, c, err)) return false;
    }
    for (size_t i = 0; i < sl.size(); ++i) {
        const uint8_t* b = src.blob(ranked[i].first, ranked[i].second);
        for (int c = 0; c < 2; ++c) {
            const OnDevice on(dev[c]);
            if (b == nullptr || !th[c].fill_staged((int32_t) i, b, nullptr, err)) {
                err = "tensor split pair: fill of slot " + std::to_string(i) + ": " + (b == nullptr ? "no blob" : err);
                return false;
            }
        }
        if ((i & 63) == 63)   // the ring is reused in stream order; keep the host from running far ahead
            for (int c = 0; c < 2; ++c) { const OnDevice on(dev[c]); cudaStreamSynchronize(nullptr); }
    }
    for (int c = 0; c < 2; ++c) {
        const OnDevice on(dev[c]);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            err = "tensor split pair: the fill failed on GPU " + std::to_string(dev[c]);
            return false;
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (size_t i : {(size_t) 0, sl.size() - 1})
        if (!th[0].verify_slot((int32_t) i, src.blob(ranked[i].first, ranked[i].second), err) ||
            !th[1].verify_slot((int32_t) i, src.blob(ranked[i].first, ranked[i].second), err))
            return false;
    uint64_t released = 0;   // in VRAM now: the arena's pages are not needed
    for (size_t i = 0; i < sl.size(); ++i) released += src.release(ranked[i].first, ranked[i].second);
    std::fprintf(stderr, "strata generate: tensor split pair: %zu experts halved (%.1f%% of %lld), 2 x %.2f GiB, "
                         "filled from the host arena in %.0f ms, %.2f GiB of arena released; first and last verified\n",
                 sl.size(), 100.0 * (double) sl.size() / (double) (g.n_layers * g.n_expert),
                 (long long) (g.n_layers * g.n_expert), (double) used / 1073741824.0, ms, (double) released / 1073741824.0);
    return true;
}

bool TpPair::init_verifiers(const ModelGeometry& g, const VerifyHits& vh, const std::vector<int32_t>& host_res,
                            const NativeHead* head, int spec, std::string& err) {
    for (int c = 0; c < 2; ++c) {
        const OnDevice on(dev[c]);
        vt[c].set_tensor_half(c, g.n_ff, !pair);
        VerifyHits vhc = vh;   // this half's arena of half experts
        vhc.cache_base = th[c].base();
        vhc.slot_off = th[c].slot_offsets();
        vhc.blob = vh.blob / 2;   // slots are sized by slot_off; this is only the "tier exists" mark
        if (dev[c] != 0) {   // the residency table on this GPU (read by the device plan only, which is off)
            int32_t* d = nullptr;
            if (cudaMalloc((void**) &d, host_res.size() * sizeof(int32_t)) != cudaSuccess ||
                cudaMemcpy(d, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "tensor split half " + std::to_string(c) + ": residency table failed";
                return false;
            }
            mem_.push_back(d);
            mem_dev_.push_back(dev[c]);
            vhc.d_res = d;
        }
        if (!vt[c].init(tw[c].table(), gh, hs[c], vhc, c == 0 ? head : nullptr, spec, err)) {
            err = "tensor split half " + std::to_string(c) + ": " + err;
            return false;
        }
        std::fprintf(stderr, "strata generate: tensor split half %d on GPU %d: %zu tensors cut, %.1f MiB; %.1f MiB "
                             "replicated\n", c, dev[c], tw[c].tensors(), (double) tw[c].bytes() / 1048576.0,
                     (double) tw[c].replicated_bytes() / 1048576.0);
    }
    return Verifier::tp_pair(vt[0], vt[1], err);
}

bool TpPair::fill(const std::vector<std::pair<int32_t, const uint8_t*>>& fills, void* stream0, std::string& err) {
    if (fills.empty()) return true;
    void* st[2] = {stream0, stream1_ != nullptr ? stream1_ : stream0};
    for (int c = 0; c < 2; ++c) { const OnDevice on(dev[c]); cudaEventRecord((cudaEvent_t) ev_[c][0], (cudaStream_t) st[c]); }
    for (const auto& [slot, b] : fills)
        for (int c = 0; c < 2; ++c) {
            const OnDevice on(dev[c]);
            if (!th[c].fill_staged(slot, b, st[c], err)) return false;
        }
    for (int c = 0; c < 2; ++c) { const OnDevice on(dev[c]); cudaEventRecord((cudaEvent_t) ev_[c][1], (cudaStream_t) st[c]); }
    in_flight_ = (int64_t) fills.size();
    return true;
}

bool TpPair::fills_done(bool wait) {
    if (in_flight_ == 0) return true;
    for (int c = 0; c < 2; ++c) {
        const OnDevice on(dev[c]);
        if (wait) cudaEventSynchronize((cudaEvent_t) ev_[c][1]);
        else if (cudaEventQuery((cudaEvent_t) ev_[c][1]) != cudaSuccess) return false;
    }
    for (int c = 0; c < 2; ++c) {
        float ms = 0;
        const OnDevice on(dev[c]);
        if (cudaEventElapsedTime(&ms, (cudaEvent_t) ev_[c][0], (cudaEvent_t) ev_[c][1]) == cudaSuccess)
            (c == 0 ? ms_half0 : ms_half1) += ms;
    }
    swapped += in_flight_;
    ++batches;
    in_flight_ = 0;
    return true;
}

bool TpPair::verify(const std::vector<std::pair<int32_t, const uint8_t*>>& fills, std::string& err) {
    for (const auto& [slot, b] : fills) {
        if (b == nullptr || !th[0].verify_slot(slot, b, err) || !th[1].verify_slot(slot, b, err)) return false;
        ++verified;
    }
    return true;
}

}  // namespace strata::core
