/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#include <iostream>
#include <cassert>

#include <map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "tensor/tuple.h"
#include "block/sched.h"

/*
template<typename T, typename U>
std::ostream& operator<<(std::ostream& os, const xpu::Tuple<T, U>& t) {
    os << "(" << t.head << ", " << t.tail << ")";
    return os;
}
*/


void test_balanced_tri_linear_sched_simple_case() {
    using namespace xpu;

    const int num_blocks_m = 4;
    const int total_tiles = (num_blocks_m * (num_blocks_m + 1)) / 2;

    // Test Case 4x4 blocks, total tiles 10
    std::vector<std::pair<int, int>> expected = {
        {0, 0},   // task_id 0
        {3, 0},   // 1
        {3, 1},   // 2
        {3, 2},   // 3
        {3, 3},   // 4
        {1, 0},   // 5
        {1, 1},   // 6
        {2, 0},   // 7
        {2, 1},   // 8
        {2, 2}    // 9
    };

    std::cout << "=== Test 1: Forward mapping (task_id -> block) ===\n";
    bool all_forward_ok = true;
    for (int tid = 0; tid < total_tiles; ++tid) {
        auto idx = get_block_indices_optimized(tid, num_blocks_m);
        int bm = xpu::get<0>(idx);
        int bn = xpu::get<1>(idx);

        auto [ebm, ebn] = expected[tid];

        bool ok = (bm == ebm && bn == ebn);

        std::cout << "task_id " << tid << " -> block(" << bm << "," << bn << ") ";
        if (ok) {
            std::cout << "✓";
        } else {
            std::cout << "✗ (expected (" << ebm << "," << ebn << "))";
        }
        std::cout << std::endl;

        if (!ok) all_forward_ok = false;
    }

    std::cout << "\n=== Test 2: Reverse mapping (block -> task_id) ===\n";
    bool all_reverse_ok = true;
    for (int tid = 0; tid < total_tiles; ++tid) {
        auto idx = get_block_indices_optimized(tid, num_blocks_m);
        int bm = xpu::get<0>(idx);
        int bn = xpu::get<1>(idx);

        int rev_tid = get_task_id_from_block(bm, bn, num_blocks_m);

        bool ok = (rev_tid == tid);

        std::cout << "block(" << bm << "," << bn << ") -> task_id " << rev_tid;
        if (ok) {
            std::cout << " ✓";
        } else {
            std::cout << " ✗ (expected " << tid << ")";
        }
        std::cout << std::endl;

        if (!ok) all_reverse_ok = false;
    }

    std::cout << "\n=== Test 3: All generated blocks cover entire lower triangular ===\n";
    std::map<std::pair<int,int>, int> coverage;
    for (int tid = 0; tid < total_tiles; ++tid) {
        auto idx = get_block_indices_optimized(tid, num_blocks_m);
        coverage[{xpu::get<0>(idx), xpu::get<1>(idx)}]++;
    }
    bool all_covered = true;
    for (int m = 0; m < num_blocks_m; ++m) {
        for (int n = 0; n <= m; ++n) {
            if (coverage.find({m,n}) == coverage.end()) {
                std::cout << "Missing block (" << m << "," << n << ")\n";
                all_covered = false;
            }
        }
    }
    if (all_covered) std::cout << "All blocks covered exactly once.\n";

    if (all_forward_ok && all_reverse_ok && all_covered)
        std::cout << "\nAll tests PASSED.\n";
    else
        std::cout << "\nSome tests FAILED.\n";
}

int main() {
    test_balanced_tri_linear_sched_simple_case();
    return 0;
}
