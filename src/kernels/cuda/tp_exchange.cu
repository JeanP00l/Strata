// src/kernels/cuda/tp_exchange.cu - see include/strata/kernels/tp_exchange.hpp.  The kernel is
// `allreduce_push` from src/kernels/p2p_bench.cu with the bench's fill and check taken out.
#include "strata/kernels/tp_exchange.hpp"

#include <cuda_runtime.h>
#if defined(STRATA_HIP_GFX906)
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <utility>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

__device__ __forceinline__ uint32_t tp_load(const uint32_t* p) {
#if defined(STRATA_HIP_GFX906)
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return *(const volatile uint32_t*) p;
#endif
}
__device__ __forceinline__ void tp_store(uint32_t* p, uint32_t v) {
#if defined(STRATA_HIP_GFX906)
    __builtin_nontemporal_store(v, p);
#else
    *(volatile uint32_t*) p = v;
#endif
}
__device__ __forceinline__ void tp_store_flag(uint32_t* p, uint32_t v) {
#if defined(STRATA_HIP_GFX906)
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    __threadfence_system();
    *(volatile uint32_t*) p = v;
#endif
}
__device__ __forceinline__ unsigned long long tp_clock() {
#if defined(STRATA_HIP_GFX906)
    return wall_clock64();
#else
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t / 40ull;   // ns -> 25 MHz ticks, the same unit as gfx906's wall clock
#endif
}

__global__ void tp_allreduce_kernel(float* x, int n, float* peer_recv, uint32_t* peer_flags, const float* recv,
                                    const uint32_t* flags, const uint32_t* seq_base, int step, long long slot_floats,
                                    int half, unsigned long long timeout, uint32_t* err) {
    const uint32_t seq = *seq_base + (uint32_t) step + 1u;
    const int slot = (int) (seq & 1u);
    const int per = (n + (int) gridDim.x - 1) / (int) gridDim.x;
    const int b0 = (int) blockIdx.x * per;
    const int b1 = min(n, b0 + per);
    float* pr = peer_recv + (size_t) slot * (size_t) slot_floats;
    for (int i = b0 + (int) threadIdx.x; i < b1; i += (int) blockDim.x) tp_store((uint32_t*) (pr + i), __float_as_uint(x[i]));
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        tp_store_flag(peer_flags + slot * kTpMaxBlocks + blockIdx.x, seq);
        const unsigned long long t0 = tp_clock();
        while (tp_load(flags + slot * kTpMaxBlocks + blockIdx.x) < seq) {
#if defined(STRATA_HIP_GFX906)
            __builtin_amdgcn_s_sleep(1);
#endif
            if (tp_clock() - t0 > timeout) {
                atomicAdd(err, 1u);
                break;
            }
        }
    }
    __syncthreads();
    const float* r = recv + (size_t) slot * (size_t) slot_floats;
    for (int i = b0 + (int) threadIdx.x; i < b1; i += (int) blockDim.x) {
        const float mine = x[i];
        const float theirs = __uint_as_float(tp_load((const uint32_t*) (r + i)));
        x[i] = half == 0 ? mine + theirs : theirs + mine;
    }
}

__global__ void tp_advance_kernel(uint32_t* seq, uint32_t by) { *seq += by; }

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void* tp_alloc_uncached(size_t bytes) {
    void* p = nullptr;
#if defined(STRATA_HIP_GFX906)
    if (hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached) != hipSuccess) return nullptr;
#else
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
#endif
    if (cudaMemset(p, 0, bytes) != cudaSuccess) { cudaFree(p); return nullptr; }
    return p;
}

void tp_free(void* p) { if (p) cudaFree(p); }

void* tp_stream_cu_half(int half) {
#if defined(STRATA_HIP_GFX906)
    int dev = 0, cus = 0;
    if (hipGetDevice(&dev) != hipSuccess ||
        hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || cus < 2)
        return nullptr;
    const int per = cus / 2, first = half == 0 ? 0 : per, last = half == 0 ? per : cus;
    uint32_t mask[8] = {};
    for (int cu = first; cu < last && cu < 256; ++cu) mask[cu / 32] |= 1u << (cu % 32);
    hipStream_t s = nullptr;
    if (hipExtStreamCreateWithCUMask(&s, (uint32_t) ((cus + 31) / 32), mask) != hipSuccess) return nullptr;
    return (void*) s;
#else
    (void) half;
    return nullptr;
#endif
}

bool tp_enable_peer(int a, int b) {
    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess) return false;
    bool ok = true;
    for (const auto& [from, to] : {std::pair<int, int>{a, b}, std::pair<int, int>{b, a}}) {
        int can = 0;
        if (cudaDeviceCanAccessPeer(&can, from, to) != cudaSuccess || !can) { ok = false; break; }
        cudaSetDevice(from);
        const cudaError_t e = cudaDeviceEnablePeerAccess(to, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ok = false;
        cudaGetLastError();
    }
    cudaSetDevice(prev);
    return ok;
}

int tp_pointer_device(const void* p) {
    if (p == nullptr) return -1;
#if defined(STRATA_HIP_GFX906)
    hipPointerAttribute_t a{};
    if (hipPointerGetAttributes(&a, p) != hipSuccess) { (void) hipGetLastError(); return -1; }
    return a.type == hipMemoryTypeDevice ? a.device : -1;
#else
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) { cudaGetLastError(); return -1; }
    return a.type == cudaMemoryTypeDevice ? a.device : -1;
#endif
}

void tp_allreduce(float* x, int64_t n, const TpChannel& ch, int step, void* stream) {
    if (n <= 0 || n > ch.slot_floats) {
        std::fprintf(stderr, "tp_allreduce: %lld floats, the channel holds %lld\n", (long long) n, (long long) ch.slot_floats);
        std::exit(1);
    }
    const int blocks = (int) std::min<int64_t>(32, std::max<int64_t>(1, n / 480));
    tp_allreduce_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(x, (int) n, ch.peer_recv, ch.peer_flags, ch.recv,
                                                                  ch.flags, ch.seq, step, (long long) ch.slot_floats,
                                                                  ch.half, (unsigned long long) ch.timeout_ticks, ch.err);
    check("tp_allreduce");
}

void tp_advance(const TpChannel& ch, uint32_t exchanges, void* stream) {
    tp_advance_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ch.seq, exchanges);
    check("tp_advance");
}

}  // namespace strata::kernels
