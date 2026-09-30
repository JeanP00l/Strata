// include/strata/kernels/tp_exchange.hpp - the tensor split's all-reduce: two halves of a layer, one on each side,
// add their partial sums inside the captured window graph.
//
// PUSH ONLY.  Each side writes its partial into the OTHER side's receive buffer and raises one flag per block
// there, then spins on ITS OWN flags for the other side's partial and adds.  Nothing reads across PCIe (the
// llama.cpp AllReduce hang on ROCm 6.3 looked like a wave waiting for a PCIe read that never came back).  Measured
// by `src/kernels/p2p_bench.cu`: 7.4-7.9 us per exchange at T = 3, 5e7 exchanges without a wrong sum.
//
// THE BUFFERS MUST BE UNCACHED (`tp_alloc_uncached`).  With fine-grained VRAM a store into the peer's memory stays
// in the WRITER's L2 until its kernel ends (gfx906 cannot flush L2 from a shader), so the flag arrives exactly when
// the writer has already given up waiting.
//
// A SPIN NEVER HANGS THE CARD.  A wait longer than `timeout_ms` (by the card's own wall clock) counts into
// `err[0]` and goes on with whatever is in the buffer; the host reads `err` after the window and refuses it loudly.
// A wrong window is an error the caller sees, not text that quietly degrades.
//
// SEQUENCING.  Exchange `step` of a window uses sequence number `*seq + step + 1`; `tp_advance` (the graph's last
// node) moves `*seq` on by the window's exchange count, so a replayed graph continues the count and the flags
// never need resetting.  Two buffer slots alternate by sequence parity: a side cannot get two exchanges ahead,
// because exchange k+1 waits for the other side's k+1, which that side writes only after it has read k.
//
// The sum is `part0 + part1` on both sides - IEEE addition of two terms commutes exactly - so both halves hold a
// bitwise identical result and every replicated computation after it (hyper-connections, router, indexer) agrees.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

inline constexpr int kTpMaxBlocks = 64;

/// One side's view of an exchange channel.  `recv`/`flags` are this side's (the other side writes them),
/// `peer_recv`/`peer_flags` the other side's.  Receive buffers hold 2 slots of `slot_floats`.
struct TpChannel {
    float* recv = nullptr;
    uint32_t* flags = nullptr;          ///< 2 slots x kTpMaxBlocks
    float* peer_recv = nullptr;
    uint32_t* peer_flags = nullptr;
    uint32_t* seq = nullptr;            ///< device, this side's exchange counter
    uint32_t* err = nullptr;            ///< device: [0] timeouts
    int64_t slot_floats = 0;
    int half = 0;                       ///< 0 or 1: which term this side's partial is
    uint64_t timeout_ticks = 0;         ///< wall-clock ticks (25 MHz on gfx906)
};

/// Device memory that bypasses the caches on every mapping (hipDeviceMallocUncached; cudaMalloc elsewhere).
void* tp_alloc_uncached(size_t bytes);
void tp_free(void* p);

/// A stream on its own hardware queue running on half `half` of the device's compute units (0: the first half),
/// for two halves sharing one GPU: two plain streams can land on one queue, and then the half launched second
/// never starts while the first spins in its exchange.  Null on failure (or where CU masks do not exist).
void* tp_stream_cu_half(int half);

/// Two halves on two GPUs: peer access both ways (already enabled counts as success).  False when either way
/// cannot be enabled.
bool tp_enable_peer(int dev_a, int dev_b);
/// The GPU a pointer's memory lives on, or -1 for host memory (pinned, mapped or pageable) and unknown pointers.
int tp_pointer_device(const void* p);

/// x (n floats, this side's partial) <- part0 + part1.  Exchange number `step` of the window.
void tp_allreduce(float* x, int64_t n, const TpChannel& ch, int step, void* stream);
/// *ch.seq += exchanges: the window's last node.
void tp_advance(const TpChannel& ch, uint32_t exchanges, void* stream);

}  // namespace strata::kernels
