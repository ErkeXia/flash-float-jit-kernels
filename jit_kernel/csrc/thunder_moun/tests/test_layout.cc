#include <iostream>
#include <cassert>

#include "tensor/tuple.h"
#include "tensor/layout.h"

#define BLOCK_M 128
#define BLOCK_N 64
#define WARP_M 32
#define WARP_N 32


void test_layout_with_heterogeneous_tuple() {
    using namespace xpu;

    // Shape< Shape<32, 128/32>, 64 >
    auto shape = make_tuple(make_tuple(Int<WARP_M>(), Int<BLOCK_M / WARP_M>()), Int<BLOCK_N>());

    // Stride< Stride<1, 32>, 128 >
    auto stride = make_tuple(make_tuple(Int<1>(), Int<WARP_M>()), Int<BLOCK_M>());

    auto first = shape[Int<0>()];
    auto second = shape[Int<1>()];

    std::cout << "[JIT Test] tuple shape : (" << first << ", " << second << ")" << std::endl;

    auto coord = make_tuple({5, 1}, 10);

    std::cout << "[JIT Test] tuple access indexer: " << coord << std::endl;

    Layout layout(shape, stride);
    auto offset = layout(coord);

    std::cout << "[JIT Test] Calculated Offset: " << offset << " (Expected: 1317)" << std::endl;
    assert(offset == 1317);

}

int main() {
    test_layout_with_heterogeneous_tuple();
    return 0;
}
