// src/core/tp_slice.cpp - see include/strata/core/tp_slice.hpp.
#include "strata/core/tp_slice.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/tp_exchange.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>

namespace strata::core {
namespace {

enum class Cut { Rows, K };
struct Plan {
    Cut cut = Cut::Rows;
    std::vector<TpRange> r;   ///< rows (Rows) or input elements (K)
};

int64_t total(const std::vector<TpRange>& rs) {
    int64_t n = 0;
    for (const auto& r : rs) n += r.end - r.begin;
    return n;
}

std::vector<TpRange> scaled(const std::vector<TpRange>& rs, int64_t by) {
    std::vector<TpRange> out;
    for (const auto& r : rs) out.push_back({r.begin * by, r.end * by});
    return out;
}

/// The cut of `blk.<l>.<suffix>` for half c, or false when the tensor is replicated.
bool plan_of(const ModelGeometry& g, int c, const std::string& name, Plan& p) {
    if (name.rfind("blk.", 0) != 0) return false;
    const size_t dot = name.find('.', 4);
    if (dot == std::string::npos) return false;
    const int64_t l = std::atoll(name.c_str() + 4);
    const std::string suffix = name.substr(dot + 1);
    const int64_t S = g.ssm_state_size;
    const int64_t NHh = g.n_head / 2, NKVh = g.n_head_kv / 2, HD = g.head_dim, FFh = g.n_ff / 2;
    if (suffix == "ffn_gate_shexp.weight" || suffix == "ffn_up_shexp.weight") { p = {Cut::Rows, {{c * FFh, (c + 1) * FFh}}}; return true; }
    if (suffix == "ffn_down_shexp.weight") { p = {Cut::K, {{c * FFh, (c + 1) * FFh}}}; return true; }
    if (is_qsa_layer(g, l)) {
        if (suffix == "attn_q.weight") { p = {Cut::Rows, {{c * NHh * 2 * HD, (c + 1) * NHh * 2 * HD}}}; return true; }
        if (suffix == "attn_k.weight" || suffix == "attn_v.weight") { p = {Cut::Rows, {{c * NKVh * HD, (c + 1) * NKVh * HD}}}; return true; }
        if (suffix == "attn_output.weight") { p = {Cut::K, {{c * NHh * HD, (c + 1) * NHh * HD}}}; return true; }
        return false;
    }
    if (suffix == "attn_qkv.weight" || suffix == "ssm_conv1d.weight") { p = {Cut::Rows, tp_gdn_channels(g, c)}; return true; }
    if (suffix == "attn_gate.weight") { p = {Cut::Rows, scaled(tp_gdn_vheads(g, c), S)}; return true; }
    if (suffix == "ssm_out.weight") { p = {Cut::K, scaled(tp_gdn_vheads(g, c), S)}; return true; }
    if (suffix == "ssm_alpha.weight" || suffix == "ssm_beta.weight" || suffix == "ssm_dt.bias" || suffix == "ssm_a") {
        p = {Cut::Rows, tp_gdn_vheads(g, c)};
        return true;
    }
    return false;
}

}  // namespace

bool tp_half_geometry(const ModelGeometry& g, ModelGeometry& h, std::string& err) {
    if (g.ssm_k_heads % 2 || g.ssm_v_heads % g.ssm_k_heads || g.n_head % 2 || g.n_head_kv % 2 || g.n_ff % 2 ||
        (g.n_head / g.n_head_kv) != 12) {
        err = "tensor split: the geometry does not halve (heads, kv heads, ff width, or a GQA group other than 12)";
        return false;
    }
    h = g;
    h.ssm_k_heads = g.ssm_k_heads / 2;
    h.ssm_v_heads = g.ssm_v_heads / 2;
    h.ssm_conv_channels = 2 * g.ssm_state_size * h.ssm_k_heads + g.ssm_state_size * h.ssm_v_heads;
    h.ssm_value_dim = g.ssm_state_size * h.ssm_v_heads;
    h.n_head = g.n_head / 2;
    h.n_head_kv = g.n_head_kv / 2;
    h.n_ff = g.n_ff / 2;
    return true;
}

std::vector<TpRange> tp_gdn_vheads(const ModelGeometry& g, int c) {
    const int64_t HK = g.ssm_k_heads, HKh = HK / 2, groups = g.ssm_v_heads / HK;
    std::vector<TpRange> out;
    for (int64_t i = 0; i < groups; ++i) out.push_back({i * HK + c * HKh, i * HK + (c + 1) * HKh});
    return out;
}

std::vector<TpRange> tp_gdn_channels(const ModelGeometry& g, int c) {
    const int64_t S = g.ssm_state_size, HK = g.ssm_k_heads, HKh = HK / 2;
    std::vector<TpRange> out;
    out.push_back({c * HKh * S, (c + 1) * HKh * S});                     // q
    out.push_back({HK * S + c * HKh * S, HK * S + (c + 1) * HKh * S});   // k
    for (const auto& v : tp_gdn_vheads(g, c)) out.push_back({2 * HK * S + v.begin * S, 2 * HK * S + v.end * S});
    return out;
}

TpWeights::~TpWeights() {
    for (void* p : allocs_) cudaFree(p);
    if (q8_1_) cudaFree(q8_1_);
}

bool TpWeights::build(const WeightTable& full, const std::vector<std::string>& shards, const ModelGeometry& g, int c,
                      std::string& err) {
    if (!allocs_.empty() || q8_1_) { err = "tensor split: half weights already built"; return false; }
    table_ = full;
    int max_in = 0;
    for (const auto& [name, ref] : full.all())
        if (ref.native_data) max_in = std::max(max_in, (int) ref.ne0);
    if (max_in <= 0) { err = "tensor split: the full table has no native projections (run with --native)"; return false; }
    if (cudaMalloc(&q8_1_, strata::kernels::native_q8_1_bytes(max_in)) != cudaSuccess) {
        err = "tensor split: q8_1 scratch allocation failed";
        return false;
    }
    auto upload = [&](const void* host, uint64_t bytes, void** dev) -> bool {
        if (cudaMalloc(dev, bytes) != cudaSuccess) { err = "tensor split: out of device memory"; return false; }
        allocs_.push_back(*dev);
        bytes_ += bytes;
        return cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    };
    std::map<std::string, bool> done;
    try {
        // ---- the native projections: GGUF block rows, cut on the host
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& t : gguf.tensors()) {
                auto it = table_.table_.find(t.name);
                if (it == table_.table_.end() || it->second.native_data == nullptr) continue;
                Plan p;
                if (!plan_of(g, c, t.name, p)) continue;
                WeightRef& ref = it->second;
                if ((int) t.type != ref.native_type || t.shape.size() != 2) {
                    err = "tensor split: " + t.name + " differs from the loaded native tensor";
                    return false;
                }
                int be = 0, bb = 0;
                if (!strata::block_geometry(t.type, be, bb)) { err = "tensor split: unknown block type " + t.name; return false; }
                const int64_t ne0 = (int64_t) t.shape[0], ne1 = (int64_t) t.shape[1];
                const uint64_t row_bytes = (uint64_t) (ne0 / be) * (uint64_t) bb;
                const auto* src = (const uint8_t*) gguf.tensor_data(t);
                std::vector<uint8_t> host;
                int64_t out0 = ne0, out1 = ne1;
                if (p.cut == Cut::Rows) {
                    out1 = total(p.r);
                    host.resize((size_t) out1 * row_bytes);
                    size_t at = 0;
                    for (const auto& r : p.r) {
                        if (r.begin < 0 || r.end > ne1) { err = "tensor split: rows out of range in " + t.name; return false; }
                        const size_t n = (size_t) (r.end - r.begin) * row_bytes;
                        std::memcpy(host.data() + at, src + (size_t) r.begin * row_bytes, n);
                        at += n;
                    }
                } else {
                    out0 = total(p.r);
                    for (const auto& r : p.r)
                        if (r.begin % be || r.end % be || r.begin < 0 || r.end > ne0) {
                            err = "tensor split: " + t.name + " cannot be cut at element " + std::to_string(r.begin) +
                                  ".." + std::to_string(r.end) + " (blocks of " + std::to_string(be) + ")";
                            return false;
                        }
                    const uint64_t out_row = (uint64_t) (out0 / be) * (uint64_t) bb;
                    host.resize((size_t) ne1 * out_row);
                    for (int64_t row = 0; row < ne1; ++row) {
                        size_t at = (size_t) row * out_row;
                        for (const auto& r : p.r) {
                            const size_t n = (size_t) ((r.end - r.begin) / be) * (size_t) bb;
                            std::memcpy(host.data() + at, src + (size_t) row * row_bytes + (size_t) (r.begin / be) * bb, n);
                            at += n;
                        }
                    }
                }
                if (host.size() != strata::kernels::native_mmvq_weight_bytes((int) t.type, (int) out0, (int) out1)) {
                    err = "tensor split: " + t.name + " cut to an unexpected byte count";
                    return false;
                }
                void* dev = nullptr;
                if (!upload(host.data(), host.size(), &dev)) { err += " (" + t.name + ")"; return false; }
                ref.native_data = dev;
                ref.native_host = false;
                ref.native_q8_1 = q8_1_;
                ref.ne0 = out0;
                ref.ne1 = out1;
                ref.elements = out0 * out1;
                done[t.name] = true;
                ++cut_;
            }
        }
    } catch (const std::exception& e) {
        err = std::string("tensor split: ") + e.what();
        return false;
    }
    // ---- the small canonical GDN tensors: rows gathered from the full arena on the device
    for (auto& [name, ref] : table_.table_) {
        if (done.count(name)) continue;
        Plan p;
        if (!plan_of(g, c, name, p)) continue;
        if (ref.native_data != nullptr) { err = "tensor split: " + name + " is native but was not in the shards"; return false; }
        if (p.cut != Cut::Rows || ref.quantized() || ref.data == nullptr) {
            err = "tensor split: " + name + " is not a resident plain tensor, so it cannot be gathered";
            return false;
        }
        const int64_t rows = ref.ne1 > 0 ? ref.ne1 : ref.ne0;
        const int64_t per_row = ref.ne1 > 0 ? ref.ne0 : 1;
        const uint64_t elem = ref.bytes / (uint64_t) (rows * per_row);
        const uint64_t row_bytes = (uint64_t) per_row * elem;
        if (elem == 0 || ref.bytes != (uint64_t) (rows * per_row) * elem) {
            err = "tensor split: " + name + " has an odd element size";
            return false;
        }
        const int64_t out_rows = total(p.r);
        void* dev = nullptr;
        if (cudaMalloc(&dev, (size_t) out_rows * row_bytes) != cudaSuccess) { err = "tensor split: out of device memory"; return false; }
        allocs_.push_back(dev);
        bytes_ += (uint64_t) out_rows * row_bytes;
        size_t at = 0;
        for (const auto& r : p.r) {
            if (r.end > rows) { err = "tensor split: rows out of range in " + name; return false; }
            const size_t n = (size_t) (r.end - r.begin) * row_bytes;
            if (cudaMemcpy((uint8_t*) dev + at, (const uint8_t*) ref.data + (size_t) r.begin * row_bytes, n,
                           cudaMemcpyDeviceToDevice) != cudaSuccess) {
                err = "tensor split: gather failed for " + name;
                return false;
            }
            at += n;
        }
        ref.data = dev;
        ref.bytes = (uint64_t) out_rows * row_bytes;
        if (ref.ne1 > 0) ref.ne1 = out_rows; else ref.ne0 = out_rows;
        ref.elements = out_rows * per_row;
        ++cut_;
    }
    // every other native projection (the PLE key) keeps the full data but uses this half's scratch
    for (auto& [name, ref] : table_.table_) {
        if (!ref.native_data || done.count(name)) continue;
        if (ref.native_host) { err = "tensor split: " + name + " is in host memory and not cut (the window would read it over PCIe)"; return false; }
        ref.native_q8_1 = q8_1_;
    }
    return true;
}

bool TpWeights::localize(std::string& err) {
    int here = 0;
    if (cudaGetDevice(&here) != cudaSuccess) { err = "tensor split: no current device"; return false; }
    auto copy = [&](const void* p, uint64_t bytes, const std::string& name) -> const void* {
        if (p == nullptr || bytes == 0) return p;
        const int dev = strata::kernels::tp_pointer_device(p);
        if (dev < 0 || dev == here) return p;
        auto it = rep_.find(p);
        if (it != rep_.end()) return it->second;
        void* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess || cudaMemcpy(d, p, bytes, cudaMemcpyDefault) != cudaSuccess) {
            cudaGetLastError();
            if (d) cudaFree(d);
            err = "tensor split: cannot replicate " + name + " (" + std::to_string(bytes >> 20) + " MiB) on GPU " +
                  std::to_string(here);
            return nullptr;
        }
        allocs_.push_back(d);
        rep_[p] = d;
        rep_bytes_ += bytes;
        return d;
    };
    for (auto& [name, ref] : table_.table_) {
        if (ref.data != nullptr) {
            const void* d = copy(ref.data, ref.bytes, name);
            if (d == nullptr) return false;
            ref.data = d;
        }
        if (ref.native_data != nullptr) {
            const void* d = copy(ref.native_data,
                                 strata::kernels::native_mmvq_weight_bytes(ref.native_type, (int) ref.ne0, (int) ref.ne1), name);
            if (d == nullptr) return false;
            ref.native_data = d;
        }
    }
    return true;
}

const void* TpWeights::local(const void* p) const {
    auto it = rep_.find(p);
    return it == rep_.end() ? p : it->second;
}

void tp_wire_half(const SessionState& f, SessionState& h, void* ple_scratch, void* ple_q8_1) {
    h.k = f.k;
    if (!f.ple.ready()) return;
    h.ple = f.ple;
    h.ple.hist = h.ple_hist;
    h.ple.token = &h.ple_token;
    h.ple.prev = h.ple_prev;
    h.ple.scratch = (float*) ple_scratch;
    if (h.ple.w.key_native_data != nullptr) h.ple.w.key_native_q8_1 = ple_q8_1;
}

bool tp_copy_state(bool to_half, const ModelGeometry& g, SessionState& f, const ModelGeometry& gh, SessionState& h,
                   int c, int64_t p0, int64_t p1, std::string& err) {
    // both sessions index qsa_states/gdn_state by global ordinal from 0: only whole-model sessions (0.1.30's layer
    // range carve is for layer-split stages, never for a tensor split's full session or its halves)
    for (const SessionState* s : {&f, &h})
        if (s->qsa_ord0 != 0 || s->gdn_ord0 != 0 || s->layer_lo != 0 || s->layer_hi != g.n_layers) {
            err = "tensor split state: a session carved for a layer range [" + std::to_string(s->layer_lo) + ", " +
                  std::to_string(s->layer_hi) + ") - the tensor split needs whole-model sessions";
            return false;
        }
    auto ok = [&](cudaError_t e, const char* what) {
        if (e == cudaSuccess) return true;
        err = std::string("tensor split state (") + what + (to_half ? ", to a half): " : ", to the full session): ") +
              cudaGetErrorString(e);
        return false;
    };
    // a copy between the full session's and the half's layout: `cp(full, half, ...)` moves full -> half or back
    auto cp2 = [&](void* full, size_t fpitch, void* half, size_t hpitch, size_t width, size_t rows, const char* what) {
        if (rows == 0 || width == 0) return true;
        return to_half ? ok(cudaMemcpy2DAsync(half, hpitch, full, fpitch, width, rows, cudaMemcpyDefault, nullptr), what)
                       : ok(cudaMemcpy2DAsync(full, fpitch, half, hpitch, width, rows, cudaMemcpyDefault, nullptr), what);
    };
    auto cp = [&](void* full, void* half, size_t bytes, const char* what) {
        return cp2(full, bytes, half, bytes, bytes, 1, what);
    };
    const bool shared = c == 0 || to_half;   // replicated state goes back from half 0 only
    const int64_t S = g.ssm_state_size;
    const uint64_t ff = (uint64_t) S * g.ssm_v_heads * S + (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const uint64_t fh = (uint64_t) S * gh.ssm_v_heads * S + (uint64_t) gh.ssm_conv_channels * (gh.ssm_d_conv - 1);
    const auto vheads = tp_gdn_vheads(g, c);
    const auto chans = tp_gdn_channels(g, c);
    const int64_t d3 = g.ssm_d_conv - 1;
    // ---- RUNNING: GDN recurrence (S, h_v, S) by value head, the conv history [C][3] by channel
    for (int64_t gi = 0; gi < g.n_gdn_layers(); ++gi) {
        float* fs = f.gdn_state + (size_t) gi * ff;
        float* hs = h.gdn_state + (size_t) gi * fh;
        int64_t at = 0;
        for (const auto& v : vheads) {
            if (!cp2(fs + v.begin * S, (size_t) g.ssm_v_heads * S * 4, hs + at * S, (size_t) gh.ssm_v_heads * S * 4,
                     (size_t) (v.end - v.begin) * S * 4, (size_t) S, "gdn state"))
                return false;
            at += v.end - v.begin;
        }
        float* fc = fs + (size_t) S * g.ssm_v_heads * S;
        float* hc = hs + (size_t) S * gh.ssm_v_heads * S;
        at = 0;
        for (const auto& r : chans) {
            if (!cp(fc + r.begin * d3, hc + at * d3, (size_t) (r.end - r.begin) * d3 * 4, "conv history")) return false;
            at += r.end - r.begin;
        }
    }
    // ---- QSA.  POSITIONAL: the kv head's rows of blocks [p0, p1), [block][kv_head][page_size][...] - from the
    // host copy of a streamed state (the authoritative one), the pool of a resident one.  The indexer: RUNNING (tail,
    // dead key, block position) whole, POSITIONAL (pooled rows) for the same blocks.
    const strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    const int64_t page = s.page_size, HD = g.head_dim;
    const int64_t NKV = g.n_head_kv, NKVh = gh.n_head_kv;
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
        QsaState& a = f.qsa_states[i];
        QsaState& b = h.qsa_states[i];
        if (a.kv_q4 || a.kv_hybrid || a.kv_int8 != b.kv_int8 || a.n_pages != b.n_pages || a.kv_mode == 2 || b.kv_mode == 2) {
            err = "tensor split: only FP16 or INT8 KV splits (not Q4 or K8V4), with the same page count";
            return false;
        }
        const int64_t b0 = std::max<int64_t>(0, p0 / page), b1 = std::min<int64_t>(a.n_pages, (p1 + page - 1) / page);
        auto rows = [&](void* fsrc, void* fhost, void* hsrc, void* hhost, uint64_t row_bytes, const char* what) {
            const size_t w = (size_t) (page * NKVh) * row_bytes, fp = (size_t) (page * NKV) * row_bytes;
            uint8_t* fb = (uint8_t*) (a.kv_mode == 1 ? fhost : fsrc);
            uint8_t* hb = (uint8_t*) (b.kv_mode == 1 ? hhost : hsrc);
            if (fb == nullptr || hb == nullptr) { err = std::string("tensor split state: no ") + what + " pool"; return false; }
            return b1 <= b0 || cp2(fb + (size_t) b0 * fp + (size_t) c * w, fp, hb + (size_t) b0 * w, w, w, (size_t) (b1 - b0), what);
        };
        if (a.kv_int8) {
            const uint64_t sc = (uint64_t) (HD / strata::kernels::KV_Q8_GROUP) * 2;
            if (!rows(a.k_q, a.host.k_q, b.k_q, b.host.k_q, (uint64_t) HD, "k") ||
                !rows(a.v_q, a.host.v_q, b.v_q, b.host.v_q, (uint64_t) HD, "v") ||
                !rows(a.k_scale, a.host.k_scale, b.k_scale, b.host.k_scale, sc, "k scale") ||
                !rows(a.v_scale, a.host.v_scale, b.v_scale, b.host.v_scale, sc, "v scale"))
                return false;
        } else if (!rows(a.k_pool, a.host.k_pool, b.k_pool, b.host.k_pool, (uint64_t) HD * 2, "k") ||
                   !rows(a.v_pool, a.host.v_pool, b.v_pool, b.host.v_pool, (uint64_t) HD * 2, "v")) {
            return false;
        }
        if (!shared) continue;
        if (a.idx_pooled_rows != b.idx_pooled_rows) { err = "tensor split: indexer sizes differ"; return false; }
        const int64_t r0 = std::max<int64_t>(0, p0 / s.idx_block);
        const int64_t r1 = std::min<int64_t>(a.idx_pooled_rows, (p1 + s.idx_block - 1) / s.idx_block + 2);
        const size_t row = (size_t) g.idx_key_dim * 4;
        if (!cp(a.idx_tail, b.idx_tail, (size_t) (s.idx_block - 1) * row, "tail") ||
            !cp(a.idx_dead, b.idx_dead, row, "dead") ||
            (r1 > r0 && !cp((uint8_t*) a.idx_pooled + (size_t) r0 * row, (uint8_t*) b.idx_pooled + (size_t) r0 * row,
                            (size_t) (r1 - r0) * row, "pooled")) ||
            !cp(a.idx_block_pos, b.idx_block_pos, 4, "block pos"))
            return false;
    }
    // ---- RUNNING, replicated: the residual, the PLE history and token window
    if (shared) {
        if (!cp(f.R, h.R, (size_t) g.hc * g.n_embd * 4, "residual")) return false;
        if (f.ple_hist != nullptr && h.ple_hist != nullptr &&
            !cp(f.ple_hist, h.ple_hist, (size_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM * 4, "ple history"))
            return false;
        SessionState& dst = to_half ? h : f;
        const SessionState& src = to_half ? f : h;
        dst.ple_prev[0] = src.ple_prev[0];
        dst.ple_prev[1] = src.ple_prev[1];
        dst.ple_token = src.ple_token;
    }
    return ok(cudaDeviceSynchronize(), "sync");
}

bool tp_split_state(const ModelGeometry& g, const SessionState& f, const ModelGeometry& gh, SessionState& h, int c,
                    void* ple_scratch, void* ple_q8_1, int64_t upto, std::string& err) {
    tp_wire_half(f, h, ple_scratch, ple_q8_1);
    return tp_copy_state(true, g, const_cast<SessionState&>(f), gh, h, c, 0, upto, err);
}

// ---------------------------------------------------------------- the routed experts' halves

std::vector<int32_t> tp_slot_layers(const std::vector<int32_t>& host_res, int64_t n_layers, int64_t n_expert,
                                    int64_t n_slots) {
    std::vector<int32_t> sl((size_t) std::max<int64_t>(n_slots, 0), -1);
    for (int64_t l = 0; l < n_layers; ++l)
        for (int64_t e = 0; e < n_expert; ++e) {
            const int32_t s = host_res[(size_t) (l * n_expert + e)];
            if (s >= 0 && s < n_slots) sl[(size_t) s] = (int32_t) l;
        }
    return sl;
}

TpExpertHalves::~TpExpertHalves() {
    if (base_) cudaFree(base_);
    if (ring_) cudaFree(ring_);
}

namespace {
struct HalfCut {   // one layer's cut, in bytes
    size_t gu = 0;           // half the gate (or up) rows
    size_t up_src = 0;       // where up starts in the full blob
    size_t down_src = 0;     // where down starts in the full blob
    size_t d_row = 0;        // a full down row
    int64_t rows = 0;        // n_embd
    size_t bytes = 0;        // the half blob
};
bool half_cut(int32_t layer, HalfCut& h) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || layer < 0 || (size_t) layer >= lay.fmt.size()) return false;
    const auto& f = lay.fmt[(size_t) layer];
    if (f.n_ff % 2 || f.d_row % 2) return false;
    h.gu = (size_t) (f.n_ff / 2) * f.gu_row;
    h.up_src = f.up_off;
    h.down_src = f.down_off;
    h.d_row = f.d_row;
    h.rows = f.n_embd;
    h.bytes = 2 * h.gu + (size_t) f.n_embd * (f.d_row / 2);
    return 2 * h.bytes == f.bytes;
}
}  // namespace

bool TpExpertHalves::open(const std::vector<int32_t>& slot_layer, int c, std::string& err) {
    if (c != 0 && c != 1) { err = "tensor split experts: the half must be 0 or 1"; return false; }
    half_ = c;
    layer_ = slot_layer;
    off_.assign(slot_layer.size() + 1, 0);
    for (size_t s = 0; s < slot_layer.size(); ++s) {
        HalfCut h;
        if (slot_layer[s] >= 0 && !half_cut(slot_layer[s], h)) {
            err = "tensor split experts: layer " + std::to_string(slot_layer[s]) +
                  " does not halve (odd ff width, odd down row, or a blob that is not [gate | up | down])";
            return false;
        }
        off_[s + 1] = off_[s] + (slot_layer[s] >= 0 ? ((uint64_t) h.bytes + 255) / 256 * 256 : 0);
    }
    if (cudaMalloc((void**) &base_, std::max<uint64_t>(off_.back(), 256)) != cudaSuccess) {
        cudaGetLastError();
        base_ = nullptr;
        err = "tensor split experts: the half arena (" + std::to_string(off_.back() >> 20) +
              " MiB) does not fit (raise --vram-reserve-mib to shrink the full cache)";
        return false;
    }
    return true;
}

bool TpExpertHalves::fill(int32_t slot, const uint8_t* full, void* stream, std::string& err) {
    HalfCut h;
    const int32_t l = layer_of(slot);
    if (l < 0 || full == nullptr || !half_cut(l, h)) { err = "tensor split experts: fill of an empty slot"; return false; }
    cudaStream_t cs = (cudaStream_t) stream;
    uint8_t* d = base_ + off_[(size_t) slot];
    const size_t hd = h.d_row / 2;
    const bool ok =
        cudaMemcpyAsync(d, full + (size_t) half_ * h.gu, h.gu, cudaMemcpyDefault, cs) == cudaSuccess &&
        cudaMemcpyAsync(d + h.gu, full + h.up_src + (size_t) half_ * h.gu, h.gu, cudaMemcpyDefault, cs) == cudaSuccess &&
        cudaMemcpy2DAsync(d + 2 * h.gu, hd, full + h.down_src + (size_t) half_ * hd, h.d_row, hd, (size_t) h.rows,
                          cudaMemcpyDefault, cs) == cudaSuccess;
    if (!ok) {
        err = std::string("tensor split experts: fill: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

bool TpExpertHalves::fill_staged(int32_t slot, const uint8_t* full_host, void* stream, std::string& err) {
    const int32_t l = layer_of(slot);
    if (l < 0 || full_host == nullptr) { err = "tensor split experts: fill of an empty slot"; return false; }
    if (ring_ == nullptr) {
        ring_blob_ = (size_t) strata::kernels::cpu::expert_layout().max_blob;
        if (cudaMalloc((void**) &ring_, (size_t) kRing * ring_blob_) != cudaSuccess) {
            cudaGetLastError();
            ring_ = nullptr;
            err = "tensor split experts: the refill ring does not fit";
            return false;
        }
    }
    const size_t bytes = (size_t) strata::kernels::cpu::expert_layout().blob_bytes(l);
    uint8_t* stage = ring_ + (size_t) ring_next_ * ring_blob_;
    ring_next_ = (ring_next_ + 1) % kRing;
    if (bytes > ring_blob_ ||
        cudaMemcpyAsync(stage, full_host, bytes, cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess) {
        err = std::string("tensor split experts: refill copy: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return fill(slot, stage, stream, err);
}

bool TpExpertHalves::verify_slot(int32_t slot, const uint8_t* full_host, std::string& err) const {
    HalfCut h;
    const int32_t l = layer_of(slot);
    if (l < 0 || !half_cut(l, h)) { err = "tensor split experts: verify of an empty slot"; return false; }
    std::vector<uint8_t> want(h.bytes), got(h.bytes);
    const size_t hd = h.d_row / 2;
    std::memcpy(want.data(), full_host + (size_t) half_ * h.gu, h.gu);
    std::memcpy(want.data() + h.gu, full_host + h.up_src + (size_t) half_ * h.gu, h.gu);
    for (int64_t r = 0; r < h.rows; ++r)
        std::memcpy(want.data() + 2 * h.gu + (size_t) r * hd, full_host + h.down_src + (size_t) r * h.d_row + (size_t) half_ * hd, hd);
    if (cudaMemcpy(got.data(), base_ + off_[(size_t) slot], h.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "tensor split experts: read-back failed";
        return false;
    }
    for (size_t i = 0; i < h.bytes; ++i)
        if (got[i] != want[i]) {
            err = "tensor split experts: half " + std::to_string(half_) + " of slot " + std::to_string(slot) +
                  " (layer " + std::to_string(l) + ") differs at byte " + std::to_string(i) + " of " + std::to_string(h.bytes);
            return false;
        }
    return true;
}

}  // namespace strata::core
