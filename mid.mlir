#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = false}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.shared = 65536 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c127_i32 = arith.constant 127 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.divsi %0, %4 : i32
    %6 = arith.subi %2, %5 : i32
    %7 = arith.minsi %6, %c1_i32 : i32
    %8 = arith.remsi %0, %4 : i32
    %9 = arith.remsi %8, %7 : i32
    %10 = arith.addi %5, %9 : i32
    %11 = arith.divsi %8, %7 : i32
    %12 = arith.muli %10, %c128_i32 : i32
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %17 = tt.splat %12 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %18 = tt.splat %12 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %19 = arith.addi %17, %13 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.addi %18, %14 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %21 = tt.splat %arg3 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.remsi %19, %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %23 = arith.muli %11, %c128_i32 : i32
    %24 = tt.splat %23 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %25 = tt.splat %23 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %24, %15 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %27 = arith.addi %25, %16 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %28 = tt.splat %arg4 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %29 = arith.remsi %27, %28 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %30 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %31 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked>
    %32 = arith.muli %30, %31 : tensor<128x1xi32, #blocked>
    %33 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %34 = tt.expand_dims %33 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %35 = tt.broadcast %32 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %36 = tt.broadcast %34 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %37 = arith.addi %35, %36 : tensor<128x128xi32, #blocked>
    %38 = tt.addptr %arg0, %c0_i32 : !tt.ptr<f16>, i32
    %39 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %40 = tt.expand_dims %39 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %41 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #blocked1>
    %42 = arith.muli %40, %41 : tensor<128x1xi32, #blocked1>
    %43 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %44 = tt.broadcast %42 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
    %45 = tt.broadcast %43 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
    %46 = arith.addi %44, %45 : tensor<128x128xi32, #blocked1>
    %47 = tt.addptr %arg1, %c0_i32 : !tt.ptr<f16>, i32
    %48 = arith.addi %arg5, %c31_i32 : i32
    %49 = arith.divsi %48, %c32_i32 : i32
    %50 = arith.muli %arg7, %c32_i32 : i32
    %51 = tt.splat %arg5 : i32 -> tensor<1x128xi32, #blocked>
    %52 = arith.cmpi slt, %34, %51 : tensor<1x128xi32, #blocked>
    %53 = tt.broadcast %52 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %54 = tt.splat %38 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %55 = tt.addptr %54, %37 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %56 = tt.load %55, %53, %cst : tensor<128x128x!tt.ptr<f16>, #blocked>
    %57 = triton_gpu.local_alloc %56 {allocation.offset = 0 : i32} : (tensor<128x128xf16, #blocked>) -> !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory, mutable>
    %58 = tt.splat %arg5 : i32 -> tensor<128x1xi32, #blocked1>
    %59 = arith.cmpi slt, %40, %58 : tensor<128x1xi32, #blocked1>
    %60 = tt.broadcast %59 : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
    %61 = tt.splat %47 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
    %62 = tt.addptr %61, %46 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
    %63 = tt.load %62, %60, %cst_0 : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %64 = triton_gpu.local_alloc %63 {allocation.offset = 32768 : i32} : (tensor<128x128xf16, #blocked1>) -> !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %65 = tt.addptr %38, %c32_i32 : !tt.ptr<f16>, i32
    %66 = tt.addptr %47, %50 : !tt.ptr<f16>, i32
    %67 = arith.subi %49, %c1_i32 : i32
    cf.br ^bb1(%c0_i32, %cst_1, %65, %66 : i32, tensor<128x128xf16, #mma>, !tt.ptr<f16>, !tt.ptr<f16>)
  ^bb1(%68: i32, %69: tensor<128x128xf16, #mma>, %70: !tt.ptr<f16>, %71: !tt.ptr<f16>):  // 2 preds: ^bb0, ^bb2
    %72 = arith.cmpi slt, %68, %67 : i32
    cf.cond_br %72, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %73 = arith.addi %68, %c1_i32 : i32
    %74 = arith.muli %73, %c32_i32 : i32
    %75 = arith.subi %arg5, %74 : i32
    %76 = tt.splat %75 : i32 -> tensor<1x128xi32, #blocked>
    %77 = arith.cmpi slt, %34, %76 : tensor<1x128xi32, #blocked>
    %78 = tt.broadcast %77 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %79 = tt.splat %70 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %80 = tt.addptr %79, %37 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %81 = tt.load %80, %78, %cst : tensor<128x128x!tt.ptr<f16>, #blocked>
    %82 = tt.splat %75 : i32 -> tensor<128x1xi32, #blocked1>
    %83 = arith.cmpi slt, %40, %82 : tensor<128x1xi32, #blocked1>
    %84 = tt.broadcast %83 : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
    %85 = tt.splat %71 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
    %86 = tt.addptr %85, %46 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
    %87 = tt.load %86, %84, %cst_0 : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %88 = triton_gpu.local_load %57 : !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %89 = triton_gpu.local_load %64 : !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %90 = tt.dot %88, %89, %69 : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf16, #mma>
    %91 = tt.addptr %70, %c32_i32 : !tt.ptr<f16>, i32
    %92 = tt.addptr %71, %50 : !tt.ptr<f16>, i32
    triton_gpu.local_store %81, %57 : tensor<128x128xf16, #blocked> -> !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %87, %64 : tensor<128x128xf16, #blocked1> -> !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %93 = arith.addi %68, %c1_i32 : i32
    cf.br ^bb1(%93, %90, %91, %92 : i32, tensor<128x128xf16, #mma>, !tt.ptr<f16>, !tt.ptr<f16>)
  ^bb3:  // pred: ^bb1
    tt.return
  }
}
