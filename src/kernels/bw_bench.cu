// src/kernels/bw_bench.cu - what a plain streaming read reaches on this card: uint4 loads, a grid-stride sum,
// for sizes from a verify window's small matrices (1-6 MB) to a large one (64 MB); plus an empty kernel's cost
// back to back (the per-launch floor a window pays ~1000 times).
//
//     build/bw_bench
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>

__global__ void read_kernel(const uint4* __restrict__ p, size_t n, float* out) {
    uint32_t acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = 1.0f;
}
__global__ void empty_kernel() {}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    uint4* buf;
    float* out;
    const size_t big = (size_t) 256 << 20;
    cudaMalloc((void**) &buf, big);
    cudaMalloc((void**) &out, 4);
    cudaMemset(buf, 1, big);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const size_t sizes_mb[] = {1, 2, 6, 16, 64, 256};
    const int blocks[] = {60, 120, 240, 480, 960, 1920};
    for (size_t mb : sizes_mb) {
        std::printf("%4zu MB:", mb);
        for (int b : blocks) {
            const size_t n = (mb << 20) / 16;
            const int it = mb >= 64 ? 20 : 200;
            // rotate through the 256 MB so small sizes do not sit in the 4 MB L2
            const size_t slots = big / (mb << 20);
            for (int i = 0; i < 5; ++i) read_kernel<<<b, 256, 0, s>>>(buf, n, out);
            cudaEventRecord(e0, s);
            for (int i = 0; i < it; ++i) read_kernel<<<b, 256, 0, s>>>(buf + (size_t) (i % slots) * n, n, out);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            std::printf("  %4d blk %5.0f GB/s", b, (double) (mb << 20) * it / (ms * 1e-3) / 1e9);
        }
        std::printf("\n");
    }
    for (int i = 0; i < 100; ++i) empty_kernel<<<1, 64, 0, s>>>();
    cudaEventRecord(e0, s);
    for (int i = 0; i < 2000; ++i) empty_kernel<<<1, 64, 0, s>>>();
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    std::printf("empty kernel back to back (stream): %.2f us each\n", 1e3 * ms / 2000);
    // the same through a graph, as the verify window runs
    cudaGraph_t g;
    cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    for (int i = 0; i < 1000; ++i) empty_kernel<<<1, 64, 0, s>>>();
    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    cudaGraphLaunch(ge, s);
    cudaStreamSynchronize(s);
    cudaEventRecord(e0, s);
    for (int i = 0; i < 5; ++i) cudaGraphLaunch(ge, s);
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    std::printf("empty kernel in a graph: %.2f us each\n", 1e3 * ms / 5000);
    return 0;
}
