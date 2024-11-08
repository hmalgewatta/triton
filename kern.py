import tempfile

import pytest
import torch
import sys

import triton
import triton.language as tl

TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8: tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz

old_kernel_string = """#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %arg3: f32 , %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %arg6: i32 {tt.divisibility = 16 : i32} , %arg7: i32 {tt.divisibility = 16 : i32} , %arg8: i32 {tt.divisibility = 16 : i32} , %arg9: i32 {tt.divisibility = 16 : i32} , %arg10: i32 {tt.divisibility = 16 : i32} , %arg11: i32 {tt.divisibility = 16 : i32} , %arg12: i32 {tt.divisibility = 16 : i32} , %arg13: i32 {tt.divisibility = 16 : i32} , %arg14: i32 {tt.divisibility = 16 : i32} , %arg15: i32 {tt.divisibility = 16 : i32} , %arg16: i32 {tt.divisibility = 16 : i32} , %arg17: i32 {tt.divisibility = 16 : i32} , %arg18: i32 {tt.divisibility = 16 : i32} , %arg19: i32 {tt.divisibility = 16 : i32} , %arg20: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma> 
    %cst_0 = arith.constant dense<0xFF800000> : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
    %c0_i64 = arith.constant 0 : i64 
    %c128_i64 = arith.constant 128 : i64 
    %c128_i32 = arith.constant 128 : i32 
    %cst_2 = arith.constant 1.44269502 : f32 
    %c0_i32 = arith.constant 0 : i32 
    %c256_i32 = arith.constant 256 : i32 
    %0 = tt.get_program_id y : i32 
    %1 = arith.muli %0, %arg7 : i32 
    %2 = tt.addptr %arg0, %1 : !tt.ptr<f16>, i32 
    %3 = tt.splat %2 : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #blocked> 
    %4 = tt.get_program_id x : i32 
    %5 = arith.muli %4, %c256_i32 : i32 
    %6 = arith.extsi %5 : i32 to i64 
    %7 = tt.splat %6 : i64 -> tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> 
    %8 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> 
    %9 = arith.extsi %8 : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> 
    %10 = arith.addi %7, %9 : tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> 
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi64, #blocked> 
    %12 = arith.extsi %arg8 : i32 to i64 
    %13 = tt.splat %12 : i64 -> tensor<256x1xi64, #blocked> 
    %14 = arith.muli %11, %13 : tensor<256x1xi64, #blocked> 
    %15 = tt.broadcast %14 : tensor<256x1xi64, #blocked> -> tensor<256x128xi64, #blocked> 
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> 
    %17 = arith.extsi %16 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>> 
    %18 = tt.expand_dims %17 {axis = 0 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi64, #blocked> 
    %19 = tt.broadcast %18 : tensor<1x128xi64, #blocked> -> tensor<256x128xi64, #blocked> 
    %20 = arith.addi %15, %19 : tensor<256x128xi64, #blocked> 
    %21 = tt.addptr %3, %20 : tensor<256x128x!tt.ptr<f16>, #blocked>, tensor<256x128xi64, #blocked> 
    %22 = tt.load %21 : tensor<256x128x!tt.ptr<f16>, #blocked> 
    %23 = tt.addptr %arg1, %1 : !tt.ptr<f16>, i32 
    %24 = arith.extsi %arg11 : i32 to i64 
    %25 = tt.addptr %arg2, %1 : !tt.ptr<f16>, i32 
    %26 = arith.extsi %arg14 : i32 to i64 
    %27 = arith.mulf %arg3, %cst_2 : f32 
    %28 = tt.splat %6 : i64 -> tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
    %29 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
    %30 = arith.extsi %29 : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>> to tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
    %31 = arith.addi %28, %30 : tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
    %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xi64, #mma> 
    %33 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>> 
    %34 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> 
    %35 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> 
    %36 = arith.extsi %33 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mma}>> 
    %37 = arith.extsi %34 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> 
    %38 = arith.extsi %35 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> 
    %39 = tt.expand_dims %36 {axis = 0 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi64, #mma> 
    %40 = tt.expand_dims %37 {axis = 0 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi64, #blocked1> 
    %41 = tt.broadcast %39 : tensor<1x128xi64, #mma> -> tensor<256x128xi64, #mma> 
    %42 = triton_gpu.convert_layout %22 : tensor<256x128xf16, #blocked> -> tensor<256x128xf16, #blocked2> 
    %43 = arith.extf %42 : tensor<256x128xf16, #blocked2> to tensor<256x128xf32, #blocked2> 
    %44 = tt.splat %27 : f32 -> tensor<256x128xf32, #blocked2> 
    %45 = arith.mulf %43, %44 : tensor<256x128xf32, #blocked2> 
    %46 = arith.truncf %45 : tensor<256x128xf32, #blocked2> to tensor<256x128xf16, #blocked2> 
    %47 = tt.splat %23 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked1> 
    %48 = tt.expand_dims %38 {axis = 1 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi64, #blocked1> 
    %49 = tt.broadcast %48 : tensor<128x1xi64, #blocked1> -> tensor<128x128xi64, #blocked1> 
    %50 = tt.splat %24 : i64 -> tensor<1x128xi64, #blocked1> 
    %51 = tt.splat %25 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked1> 
    %52 = tt.splat %26 : i64 -> tensor<1x128xi64, #blocked1> 
    %53 = arith.muli %40, %52 : tensor<1x128xi64, #blocked1> 
    %54 = tt.broadcast %53 : tensor<1x128xi64, #blocked1> -> tensor<128x128xi64, #blocked1> 
    %55:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %cst, %arg23 = %cst_1, %arg24 = %cst_0, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (tensor<256x128xf32, #mma>, tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, i64, i64)  : i32 {
        %68 = tt.splat %arg26 : i64 -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> 
        %69 = arith.addi %68, %37 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> 
        %70 = tt.expand_dims %69 {axis = 0 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi64, #blocked1> 
        %71 = arith.muli %70, %50 : tensor<1x128xi64, #blocked1> 
        %72 = tt.broadcast %71 : tensor<1x128xi64, #blocked1> -> tensor<128x128xi64, #blocked1> 
        %73 = arith.addi %49, %72 : tensor<128x128xi64, #blocked1> 
        %74 = tt.addptr %47, %73 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi64, #blocked1> 
        %75 = tt.load %74 : tensor<128x128x!tt.ptr<f16>, #blocked1> 
        %76 = tt.splat %arg25 : i64 -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> 
        %77 = arith.addi %76, %38 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> 
        %78 = tt.expand_dims %77 {axis = 1 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi64, #blocked1> 
        %79 = tt.broadcast %78 : tensor<128x1xi64, #blocked1> -> tensor<128x128xi64, #blocked1> 
        %80 = arith.addi %79, %54 : tensor<128x128xi64, #blocked1> 
        %81 = tt.addptr %51, %80 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi64, #blocked1> 
        %82 = tt.load %81 : tensor<128x128x!tt.ptr<f16>, #blocked1> 
        %83 = triton_gpu.convert_layout %75 : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #blocked3> 
        %84 = amdgpu.view_slice %46[0, 0] [256, 32] [1, 1] : tensor<256x128xf16, #blocked2> to tensor<256x32xf16, #blocked2> 
        %85 = triton_gpu.local_alloc %84 : (tensor<256x32xf16, #blocked2>) -> !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> 
        %86 = triton_gpu.local_load %85 : !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
        %87 = amdgpu.view_slice %83[0, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked3> to tensor<32x128xf16, #blocked3> 
        %88 = triton_gpu.local_alloc %87 : (tensor<32x128xf16, #blocked3>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
        %89 = triton_gpu.local_load %88 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
        %91 = amdgpu.view_slice %46[0, 32] [256, 32] [1, 1] : tensor<256x128xf16, #blocked2> to tensor<256x32xf16, #blocked2> 
        %92 = triton_gpu.local_alloc %91 : (tensor<256x32xf16, #blocked2>) -> !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> 
        %93 = triton_gpu.local_load %92 : !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
        %94 = amdgpu.view_slice %83[32, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked3> to tensor<32x128xf16, #blocked3> 
        %90 = tt.dot %86, %89, %cst : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
        %95 = triton_gpu.local_alloc %94 : (tensor<32x128xf16, #blocked3>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
        %96 = triton_gpu.local_load %95 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
        %98 = amdgpu.view_slice %46[0, 64] [256, 32] [1, 1] : tensor<256x128xf16, #blocked2> to tensor<256x32xf16, #blocked2> 
        %99 = triton_gpu.local_alloc %98 : (tensor<256x32xf16, #blocked2>) -> !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> 
        %100 = triton_gpu.local_load %99 : !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
        %101 = amdgpu.view_slice %83[64, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked3> to tensor<32x128xf16, #blocked3> 
        %102 = triton_gpu.local_alloc %101 : (tensor<32x128xf16, #blocked3>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
        %103 = triton_gpu.local_load %102 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
        %97 = tt.dot %93, %96, %90 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
        %105 = amdgpu.view_slice %46[0, 96] [256, 32] [1, 1] : tensor<256x128xf16, #blocked2> to tensor<256x32xf16, #blocked2> 
        %106 = triton_gpu.local_alloc %105 : (tensor<256x32xf16, #blocked2>) -> !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> 
        %107 = triton_gpu.local_load %106 : !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
        %108 = amdgpu.view_slice %83[96, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked3> to tensor<32x128xf16, #blocked3> 
        %109 = triton_gpu.local_alloc %108 : (tensor<32x128xf16, #blocked3>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
        %110 = triton_gpu.local_load %109 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
        %104 = tt.dot %100, %103, %97 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
        %111 = tt.dot %107, %110, %104 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
        %112 = "tt.reduce"(%111) <{axis = 1 : i32}> ({
        ^bb0(%arg27: f32 , %arg28: f32 ):
            %154 = arith.maxnumf %arg27, %arg28 : f32 
            tt.reduce.return %154 : f32 
        }) : (tensor<256x128xf32, #mma>) -> tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
        %113 = arith.maxnumf %arg24, %112 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
        %114 = tt.expand_dims %113 {axis = 1 : i32} : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xf32, #mma> 
        %115 = tt.broadcast %114 : tensor<256x1xf32, #mma> -> tensor<256x128xf32, #mma> 
        %116 = arith.subf %111, %115 : tensor<256x128xf32, #mma> 
        %117 = math.exp2 %116 : tensor<256x128xf32, #mma> 
        %118 = arith.subf %arg24, %113 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
        %119 = math.exp2 %118 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
        %120 = tt.expand_dims %119 {axis = 1 : i32} : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xf32, #mma> 
        %121 = tt.broadcast %120 : tensor<256x1xf32, #mma> -> tensor<256x128xf32, #mma> 
        %122 = arith.mulf %arg22, %121 : tensor<256x128xf32, #mma> 
        %123 = arith.truncf %117 : tensor<256x128xf32, #mma> to tensor<256x128xf16, #mma> 
        %124 = triton_gpu.convert_layout %82 : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #blocked3> 
        %125 = amdgpu.view_slice %123[0, 0] [256, 32] [1, 1] : tensor<256x128xf16, #mma> to tensor<256x32xf16, #mma> 
        %126 = triton_gpu.convert_layout %125 : tensor<256x32xf16, #mma> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
        %127 = amdgpu.view_slice %124[0, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked3> to tensor<32x128xf16, #blocked3> 
        %128 = triton_gpu.local_alloc %127 : (tensor<32x128xf16, #blocked3>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
        %129 = triton_gpu.local_load %128 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
         %131 = amdgpu.view_slice %123[0, 32] [256, 32] [1, 1] : tensor<256x128xf16, #mma> to tensor<256x32xf16, #mma> 
        %132 = triton_gpu.convert_layout %131 : tensor<256x32xf16, #mma> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
        %133 = amdgpu.view_slice %124[32, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked3> to tensor<32x128xf16, #blocked3> 
        %134 = triton_gpu.local_alloc %133 : (tensor<32x128xf16, #blocked3>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
        %135 = triton_gpu.local_load %134 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
        %130 = tt.dot %126, %129, %122 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
        %137 = amdgpu.view_slice %123[0, 64] [256, 32] [1, 1] : tensor<256x128xf16, #mma> to tensor<256x32xf16, #mma> 
        %138 = triton_gpu.convert_layout %137 : tensor<256x32xf16, #mma> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
        %139 = amdgpu.view_slice %124[64, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked3> to tensor<32x128xf16, #blocked3> 
        %140 = triton_gpu.local_alloc %139 : (tensor<32x128xf16, #blocked3>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
        %141 = triton_gpu.local_load %140 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
        %136 = tt.dot %132, %135, %130 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
        %143 = amdgpu.view_slice %123[0, 96] [256, 32] [1, 1] : tensor<256x128xf16, #mma> to tensor<256x32xf16, #mma> 
        %144 = triton_gpu.convert_layout %143 : tensor<256x32xf16, #mma> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
        %145 = amdgpu.view_slice %124[96, 0] [32, 128] [1, 1] : tensor<128x128xf16, #blocked3> to tensor<32x128xf16, #blocked3> 
        %146 = triton_gpu.local_alloc %145 : (tensor<32x128xf16, #blocked3>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
        %147 = triton_gpu.local_load %146 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
        %142 = tt.dot %138, %141, %136 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
        %148 = tt.dot %144, %147, %142 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
        %149 = "tt.reduce"(%117) <{axis = 1 : i32}> ({
        ^bb0(%arg27: f32 , %arg28: f32 ):
            %154 = arith.addf %arg27, %arg28 : f32 
            tt.reduce.return %154 : f32 
        }) : (tensor<256x128xf32, #mma>) -> tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
        %150 = arith.mulf %arg23, %119 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
        %151 = arith.addf %150, %149 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
        %152 = arith.addi %arg25, %c128_i64 : i64 
        %153 = arith.addi %arg26, %c128_i64 : i64 
        scf.yield %148, %151, %113, %152, %153 : tensor<256x128xf32, #mma>, tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, i64, i64 
        } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>} 
    %56 = tt.expand_dims %55#1 {axis = 1 : i32} : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xf32, #mma> 
    %57 = tt.broadcast %56 : tensor<256x1xf32, #mma> -> tensor<256x128xf32, #mma> 
    %58 = arith.divf %55#0, %57 : tensor<256x128xf32, #mma> 
    %59 = tt.addptr %arg5, %1 : !tt.ptr<f16>, i32 
    %60 = arith.extsi %arg17 : i32 to i64 
    %61 = arith.truncf %58 : tensor<256x128xf32, #mma> to tensor<256x128xf16, #mma> 
    %62 = tt.splat %59 : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #mma> 
    %63 = tt.splat %60 : i64 -> tensor<256x1xi64, #mma> 
    %64 = arith.muli %32, %63 : tensor<256x1xi64, #mma> 
    %65 = tt.broadcast %64 : tensor<256x1xi64, #mma> -> tensor<256x128xi64, #mma> 
    %66 = arith.addi %65, %41 : tensor<256x128xi64, #mma> 
    %67 = tt.addptr %62, %66 : tensor<256x128x!tt.ptr<f16>, #mma>, tensor<256x128xi64, #mma> 
    tt.store %67, %61 : tensor<256x128x!tt.ptr<f16>, #mma> 
    tt.return 
} 
} 
"""

kernel_string = """#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 8], order = [1, 0]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %arg3: f32 , %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %arg6: i32 {tt.divisibility = 16 : i32} , %arg7: i32 {tt.divisibility = 16 : i32} , %arg8: i32 {tt.divisibility = 16 : i32} , %arg9: i32 {tt.divisibility = 16 : i32} , %arg10: i32 {tt.divisibility = 16 : i32} , %arg11: i32 {tt.divisibility = 16 : i32} , %arg12: i32 {tt.divisibility = 16 : i32} , %arg13: i32 {tt.divisibility = 16 : i32} , %arg14: i32 {tt.divisibility = 16 : i32} , %arg15: i32 {tt.divisibility = 16 : i32} , %arg16: i32 {tt.divisibility = 16 : i32} , %arg17: i32 {tt.divisibility = 16 : i32} , %arg18: i32 , %arg19: i32 {tt.divisibility = 16 : i32} , %arg20: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c256_i32 = arith.constant 256 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %cst = arith.constant 1.44269502 : f32 
    %c128_i32 = arith.constant 128 : i32 
    %c128_i64 = arith.constant 128 : i64 
    %c0_i64 = arith.constant 0 : i64 
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
    %cst_1 = arith.constant dense<0xFF800000> : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma> 
    %0 = tt.get_program_id x : i32 
    %1 = tt.get_program_id y : i32 
    %2 = arith.muli %1, %arg7 : i32 
    %3 = tt.addptr %arg0, %2 : !tt.ptr<f16>, i32 
    %4 = arith.muli %0, %c256_i32 : i32 
    %5 = arith.extsi %arg8 : i32 to i64 
    %6 = arith.extsi %4 : i32 to i64 
    %7 = tt.addptr %arg1, %2 : !tt.ptr<f16>, i32 
    %8 = arith.extsi %arg11 : i32 to i64 
    %9 = tt.addptr %arg2, %2 : !tt.ptr<f16>, i32 
    %10 = arith.extsi %arg14 : i32 to i64 
    %11 = arith.mulf %arg3, %cst : f32 
    %12 = tt.splat %3 : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #blocked> 
    %13 = tt.splat %6 : i64 -> tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> 
    %14 = tt.splat %6 : i64 -> tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
    %15 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> 
    %16 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
    %17 = arith.extsi %15 : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> 
    %18 = arith.extsi %16 : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>> to tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
    %19 = arith.addi %13, %17 : tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> 
    %20 = arith.addi %14, %18 : tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
    %21 = tt.expand_dims %19 {axis = 1 : i32} : tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi64, #blocked> 
    %22 = tt.expand_dims %20 {axis = 1 : i32} : tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xi64, #mma> 
    %23 = tt.splat %5 : i64 -> tensor<256x1xi64, #blocked> 
    %24 = arith.muli %21, %23 : tensor<256x1xi64, #blocked> 
    %25 = tt.broadcast %24 : tensor<256x1xi64, #blocked> -> tensor<256x128xi64, #blocked> 
    %26 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> 
    %27 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>> 
    %28 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> 
    %29 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> 
    %30 = arith.extsi %26 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>> 
    %31 = arith.extsi %27 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mma}>> 
    %32 = arith.extsi %28 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> 
    %33 = arith.extsi %29 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> 
    %34 = tt.expand_dims %30 {axis = 0 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi64, #blocked> 
    %35 = tt.expand_dims %31 {axis = 0 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi64, #mma> 
    %36 = tt.expand_dims %32 {axis = 0 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi64, #blocked1> 
    %37 = tt.broadcast %34 : tensor<1x128xi64, #blocked> -> tensor<256x128xi64, #blocked> 
    %38 = tt.broadcast %35 : tensor<1x128xi64, #mma> -> tensor<256x128xi64, #mma> 
    %39 = arith.addi %25, %37 : tensor<256x128xi64, #blocked> 
    %40 = tt.addptr %12, %39 : tensor<256x128x!tt.ptr<f16>, #blocked>, tensor<256x128xi64, #blocked> 
    %41 = tt.splat %7 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked1> 
    %42 = tt.expand_dims %33 {axis = 1 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi64, #blocked1> 
    %43 = tt.broadcast %42 : tensor<128x1xi64, #blocked1> -> tensor<128x128xi64, #blocked1> 
    %44 = tt.splat %8 : i64 -> tensor<1x128xi64, #blocked1> 
    %45 = tt.splat %11 : f32 -> tensor<256x128xf32, #mma> 
    %46 = tt.splat %9 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked1> 
    %47 = tt.splat %10 : i64 -> tensor<1x128xi64, #blocked1> 
    %48 = arith.muli %36, %47 : tensor<1x128xi64, #blocked1> 
    %49 = tt.broadcast %48 : tensor<1x128xi64, #blocked1> -> tensor<128x128xi64, #blocked1> 
    %74 = amdgpu.view_slice %40[0, 32] [256, 32] [1, 1] : tensor<256x128x!tt.ptr<f16>, #blocked> to tensor<256x32x!tt.ptr<f16>, #blocked> 
    %75 = tt.load %74 : tensor<256x32x!tt.ptr<f16>, #blocked> 
    %63 = amdgpu.view_slice %40[0, 0] [256, 32] [1, 1] : tensor<256x128x!tt.ptr<f16>, #blocked> to tensor<256x32x!tt.ptr<f16>, #blocked> 
    %64 = tt.load %63 : tensor<256x32x!tt.ptr<f16>, #blocked> 
    %78 = amdgpu.view_slice %40[0, 64] [256, 32] [1, 1] : tensor<256x128x!tt.ptr<f16>, #blocked> to tensor<256x32x!tt.ptr<f16>, #blocked> 
    %79 = tt.load %78 : tensor<256x32x!tt.ptr<f16>, #blocked> 
    %82 = amdgpu.view_slice %40[0, 96] [256, 32] [1, 1] : tensor<256x128x!tt.ptr<f16>, #blocked> to tensor<256x32x!tt.ptr<f16>, #blocked> 
    %83 = tt.load %82 : tensor<256x32x!tt.ptr<f16>, #blocked> 
      %50:5 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg22 = %cst_2, %arg23 = %cst_0, %arg24 = %cst_1, %arg25 = %c0_i64, %arg26 = %c0_i64) -> (tensor<256x128xf32, #mma>, tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, i64, i64)  : i32 {
      %65 = tt.splat %arg26 : i64 -> tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> 
      %66 = arith.addi %65, %32 : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> 
      %67 = tt.expand_dims %66 {axis = 0 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi64, #blocked1> 
      %68 = arith.muli %67, %44 : tensor<1x128xi64, #blocked1> 
      %69 = tt.broadcast %68 : tensor<1x128xi64, #blocked1> -> tensor<128x128xi64, #blocked1> 
      %70 = arith.addi %43, %69 : tensor<128x128xi64, #blocked1> 
      %71 = tt.addptr %41, %70 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi64, #blocked1> 
      %72 = amdgpu.view_slice %71[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16>, #blocked1> to tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %73 = tt.load %72 : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %76 = amdgpu.view_slice %71[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16>, #blocked1> to tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %77 = tt.load %76 : tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %100 = triton_gpu.local_alloc %64 : (tensor<256x32xf16, #blocked>) -> !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> 
      %101 = triton_gpu.local_load %100 : !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
      %102 = triton_gpu.local_alloc %73 : (tensor<32x128xf16, #blocked1>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
      %103 = triton_gpu.local_load %102 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
      %104 = tt.dot %101, %103, %cst_2 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
      amdgpu.instruction_sched_hint 
      %105 = triton_gpu.local_alloc %75 : (tensor<256x32xf16, #blocked>) -> !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> 
      %106 = triton_gpu.local_load %105 : !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
      %107 = triton_gpu.local_alloc %77 : (tensor<32x128xf16, #blocked1>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
      %108 = triton_gpu.local_load %107 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
      %80 = amdgpu.view_slice %71[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16>, #blocked1> to tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %81 = tt.load %80 : tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %109 = tt.dot %106, %108, %104 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
      amdgpu.instruction_sched_hint 
      %84 = amdgpu.view_slice %71[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16>, #blocked1> to tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %85 = tt.load %84 : tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %110 = triton_gpu.local_alloc %79 : (tensor<256x32xf16, #blocked>) -> !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> 
      %111 = triton_gpu.local_load %110 : !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
      %112 = triton_gpu.local_alloc %81 : (tensor<32x128xf16, #blocked1>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
      %113 = triton_gpu.local_load %112 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
      %114 = tt.dot %111, %113, %109 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
      amdgpu.instruction_sched_hint 
      %115 = triton_gpu.local_alloc %83 : (tensor<256x32xf16, #blocked>) -> !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> 
      %116 = triton_gpu.local_load %115 : !tt.memdesc<256x32xf16, #shared, #triton_gpu.shared_memory> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
      %117 = triton_gpu.local_alloc %85 : (tensor<32x128xf16, #blocked1>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
      %118 = triton_gpu.local_load %117 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
      %86 = tt.splat %arg25 : i64 -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> 
      %87 = arith.addi %86, %33 : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> 
      %88 = tt.expand_dims %87 {axis = 1 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi64, #blocked1> 
      %89 = tt.broadcast %88 : tensor<128x1xi64, #blocked1> -> tensor<128x128xi64, #blocked1> 
      %90 = arith.addi %89, %49 : tensor<128x128xi64, #blocked1> 
      %91 = tt.addptr %46, %90 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi64, #blocked1> 
      %92 = amdgpu.view_slice %91[0, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16>, #blocked1> to tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %93 = tt.load %92 : tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %119 = tt.dot %116, %118, %114 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
      amdgpu.instruction_sched_hint 
      %94 = amdgpu.view_slice %91[32, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16>, #blocked1> to tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %95 = tt.load %94 : tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %96 = amdgpu.view_slice %91[64, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16>, #blocked1> to tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %97 = tt.load %96 : tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %120 = arith.mulf %119, %45 : tensor<256x128xf32, #mma> 
      %121 = "tt.reduce"(%120) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32 , %arg28: f32 ):
        %158 = arith.maxnumf %arg27, %arg28 : f32 
        tt.reduce.return %158 : f32 
      }) : (tensor<256x128xf32, #mma>) -> tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
      %122 = arith.maxnumf %arg24, %121 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
      %123 = tt.expand_dims %122 {axis = 1 : i32} : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xf32, #mma> 
      %124 = tt.broadcast %123 : tensor<256x1xf32, #mma> -> tensor<256x128xf32, #mma> 
      %125 = arith.subf %120, %124 : tensor<256x128xf32, #mma> 
      %126 = math.exp2 %125 : tensor<256x128xf32, #mma> 
      %127 = arith.subf %arg24, %122 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
      %128 = math.exp2 %127 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
      %129 = tt.expand_dims %128 {axis = 1 : i32} : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xf32, #mma> 
      %130 = tt.broadcast %129 : tensor<256x1xf32, #mma> -> tensor<256x128xf32, #mma> 
      %131 = arith.mulf %arg22, %130 : tensor<256x128xf32, #mma> 
      %132 = arith.truncf %126 : tensor<256x128xf32, #mma> to tensor<256x128xf16, #mma> 
      %133 = amdgpu.view_slice %132[0, 0] [256, 32] [1, 1] : tensor<256x128xf16, #mma> to tensor<256x32xf16, #mma> 
      %134 = triton_gpu.convert_layout %133 : tensor<256x32xf16, #mma> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
      %135 = triton_gpu.local_alloc %93 : (tensor<32x128xf16, #blocked1>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
      %136 = triton_gpu.local_load %135 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
      %137 = tt.dot %134, %136, %131 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
      amdgpu.instruction_sched_hint 
      %98 = amdgpu.view_slice %91[96, 0] [32, 128] [1, 1] : tensor<128x128x!tt.ptr<f16>, #blocked1> to tensor<32x128x!tt.ptr<f16>, #blocked1> 
      %99 = tt.load %98 : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %138 = amdgpu.view_slice %132[0, 32] [256, 32] [1, 1] : tensor<256x128xf16, #mma> to tensor<256x32xf16, #mma> 
      %139 = triton_gpu.convert_layout %138 : tensor<256x32xf16, #mma> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
      %140 = triton_gpu.local_alloc %95 : (tensor<32x128xf16, #blocked1>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
      %141 = triton_gpu.local_load %140 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
      %142 = tt.dot %139, %141, %137 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
      amdgpu.instruction_sched_hint 
      %143 = amdgpu.view_slice %132[0, 64] [256, 32] [1, 1] : tensor<256x128xf16, #mma> to tensor<256x32xf16, #mma> 
      %144 = triton_gpu.convert_layout %143 : tensor<256x32xf16, #mma> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
      %145 = triton_gpu.local_alloc %97 : (tensor<32x128xf16, #blocked1>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
      %146 = triton_gpu.local_load %145 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
      %147 = tt.dot %144, %146, %142 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
      amdgpu.instruction_sched_hint 
      %148 = amdgpu.view_slice %132[0, 96] [256, 32] [1, 1] : tensor<256x128xf16, #mma> to tensor<256x32xf16, #mma> 
      %149 = triton_gpu.convert_layout %148 : tensor<256x32xf16, #mma> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
      %150 = triton_gpu.local_alloc %99 : (tensor<32x128xf16, #blocked1>) -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> 
      %151 = triton_gpu.local_load %150 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
      %152 = tt.dot %149, %151, %147 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma> 
      amdgpu.instruction_sched_hint 
      %153 = "tt.reduce"(%126) <{axis = 1 : i32}> ({
      ^bb0(%arg27: f32 , %arg28: f32 ):
        %158 = arith.addf %arg27, %arg28 : f32 
        tt.reduce.return %158 : f32 
      }) : (tensor<256x128xf32, #mma>) -> tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
      %154 = arith.mulf %arg23, %128 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
      %155 = arith.addf %154, %153 : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> 
      %156 = arith.addi %arg25, %c128_i64 : i64 
      %157 = arith.addi %arg26, %c128_i64 : i64 
      scf.yield %152, %155, %122, %156, %157 : tensor<256x128xf32, #mma>, tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, i64, i64 
    } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>} 
    %51 = tt.expand_dims %50#1 {axis = 1 : i32} : tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xf32, #mma> 
    %52 = tt.broadcast %51 : tensor<256x1xf32, #mma> -> tensor<256x128xf32, #mma> 
    %53 = arith.divf %50#0, %52 : tensor<256x128xf32, #mma> 
    %54 = tt.addptr %arg5, %2 : !tt.ptr<f16>, i32 
    %55 = arith.extsi %arg17 : i32 to i64 
    %56 = arith.truncf %53 : tensor<256x128xf32, #mma> to tensor<256x128xf16, #mma> 
    %57 = tt.splat %54 : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #mma> 
    %58 = tt.splat %55 : i64 -> tensor<256x1xi64, #mma> 
    %59 = arith.muli %22, %58 : tensor<256x1xi64, #mma> 
    %60 = tt.broadcast %59 : tensor<256x1xi64, #mma> -> tensor<256x128xi64, #mma> 
    %61 = arith.addi %60, %38 : tensor<256x128xi64, #mma> 
    %62 = tt.addptr %57, %61 : tensor<256x128x!tt.ptr<f16>, #mma>, tensor<256x128xi64, #mma> 
    tt.store %62, %56 : tensor<256x128x!tt.ptr<f16>, #mma> 
    tt.return 
  } 
} 
 """
with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
    f.write(kernel_string)
    f.flush()
    kernel = triton.compile(f.name)

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-2]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q, dtype=v.dtype)
        if torch.version.hip is None:
            BLOCK_M = 128
            BLOCK_N = 64 if Lk <= 64 else 32
            num_stages = 4 if Lk <= 64 else 3
            num_warps = 4 if Lk <= 64 else 8

        ## hardcoded best perf_configs for MI250
        if Lk == 64:
            ## D_HEAD = 64
            BLOCK_M = 128
            BLOCK_N = 64
            waves_per_eu = 3
            num_warps = 4
            num_stages = 1
            ## causal=False likes to pre load v but causal=True does not
            pre_load_v = False if causal else True
            slice_k_tile = 32
            kpack = 1
        else:
            ## D_HEAD = 128
            ## For fp16, pick BLOCK_M=256, num_warps=8
            ## For fp8, pick BLOCK_M=128, num_warps=4
            ## TODO (zhanglx): add tuning infra for FA
            BLOCK_M = 128 if TORCH_HAS_FP8E4 and q.dtype == torch.float8_e4m3fnuz else 256
            BLOCK_N = 128
            waves_per_eu = 2
            num_warps = BLOCK_M // 32
            num_stages = 1
            pre_load_v = False
            slice_k_tile = 32
            kpack = 1

        

        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        # arg20: N_CTX
        # arg19: not used
        # arg18: nu
        # arg17
        # arg3: sm_scale
        # arg4: nu
        # arg5: Out
        # arg6: not used
        # arg7: stride_qh == q.stride(1)
        # arg8: stride_qm
        # arg9: nu
        # arg10: nu
        # arg11: idk (order K?) probably stride_kn
        # arg12: nu
        # arg13: nu
        # arg14: idk (order V?) probably stride_vn
        # arg15: nu
        # arg16: nu
        # arg17: idk blockpinter (order OutBlockptr) stride_om
        # arg18: nu
        # arg18: nu
        # arg20: N_CTX

        kernel[grid](
            q,
            k,
            v,
            sm_scale,
            M,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            q.shape[2],
            q.shape[2],
            q.shape[2],
        )

        return o

attention = _attention.apply

name_to_torch_types = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp8': float8}


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD, dtype',
                         [(*shape, dtype)
                          for shape in [
                            (16, 16, 1024, 128),
                            # (8, 16, 2048, 128),
                            # (4, 16, 4096, 128),
                            # (2, 16, 8192, 128),
                            # (1, 16, 16384, 128),
                            # (4, 48, 1024, 128),
                            # (4, 48, 2048, 128),
                            # (4, 48, 4096, 128),
                            # (4, 48, 8192, 128),
                            # (4, 48, 16384, 128),
                            ]
                          for dtype in ['fp16']])
def test_op_fwd(Z, H, N_CTX, D_HEAD, dtype):
    torch.manual_seed(20)
    init_dtype = torch.float16 if dtype == 'fp8' else name_to_torch_types[dtype]
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_())
    v = (torch.empty((Z, H, D_HEAD, N_CTX), dtype=init_dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_())
    sm_scale = 0.5
    # reference implementation
    # M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    ref_out = torch.matmul(p, v.transpose(2, 3))
    # triton implementation
    # q,k casting for partial fp8
    q = q.to(name_to_torch_types[dtype])
    k = k.to(name_to_torch_types[dtype])
    # dout = torch.randn_like(q, dtype=torch.float16)
    tri_out = attention(q, k, v, sm_scale)
    # compare
    atol = 1.4e-1 if dtype == 'fp8' else 1e-2
    rtol = 1e-2 if dtype == 'fp8' else 3e-3
    torch.testing.assert_close(ref_out, tri_out, atol=atol, rtol=rtol)


try:
    FLASH_VER = 2
except BaseException:
    try:
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None

# vary seq length for fixed head and batch=4
configs = []
for dtype in ['fp16']:
    for D_HEAD in [128]:
        for causal in [False]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=['BATCH', 'H', 'N_CTX'], x_vals=[
                        # (16, 16, 1024),
                        # (8, 16, 2048),
                        # (4, 16, 4096),
                        # (2, 16, 8192),
                        # (1, 16, 16384),
                        # (4, 48, 1024),
                        # (4, 48, 2048),
                        # (4, 48, 4096),
                        # (4, 48, 8192),
                        (4, 48, 16384),
                    ], line_arg='provider', line_vals=['triton'], line_names=['Triton'],
                    #styles=[('red', '-'), ('blue', '-')],
                    ylabel='ms', plot_name=f'fused-attention-fwd-d{D_HEAD}-causal={causal}-{dtype}',
                    args={'D_HEAD': D_HEAD, 'dtype': dtype, 'causal': causal}))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, provider, dtype, device="cuda"):
    if dtype == 'fp8' and not TORCH_HAS_FP8E4:
        sys.exit("fp8 is not available")
    warmup = 25
    rep = 100
    init_dtype = torch.float16 if dtype != 'bf16' else torch.bfloat16
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, H, D_HEAD, N_CTX), dtype=init_dtype, device="cuda", requires_grad=True)
    sm_scale = 1.3
    # q,k casting for partial fp8
    q = q.to(name_to_torch_types[dtype])
    k = k.to(name_to_torch_types[dtype])
    fn = lambda: attention(q, k, v, sm_scale)
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    return total_flops / ms * 1e-9


def main():
    bench_flash_attention.run(save_path='.', print_data=True)


if __name__ == '__main__':
    sys.exit(main())
