/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once
#include "../tensor/tensor_view_ref.h"

namespace xpu {

template <typename T, int M, int N>
struct alignas(128) SharedBlock {
    T data[M][N];
};

struct BlockLaunchParams {
    int block_m;
    int block_n;
    int block_k;
    int split_k_slices;
};

} // namespace xpu
