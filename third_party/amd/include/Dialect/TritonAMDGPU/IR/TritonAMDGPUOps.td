/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#ifndef TRITON_AMDGPU_OPS
#define TRITON_AMDGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "TritonAMDGPUDialect.td"
include "TritonAMDGPUAttrDefs.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ViewLikeInterface.td" // OffsetSizeAndStrideOpInterface

class TritonAMDGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonAMDGPU_Dialect, mnemonic, traits>;

def TritonAMDGPU_ViewSliceOp : TritonAMDGPU_Op<"view_slice",
                                [AttrSizedOperandSegments,
                                 Pure,
                                 OffsetSizeAndStrideOpInterface
                                 ]> {
  let summary = "view slice operation";
  let description = [{
    Represents view of the slice of the tensor in registers. However, unlike 'memdesc_subview' which provides a view in shared memory,
    'view_slice' specifically slices within registers.
    Slice of the tensor is required to have the same layout as the original tensor.
    }];

  let arguments = (ins
    AnyRankedTensor:$source,
    Variadic<I32>:$offsets,
    Variadic<I32>:$sizes,
    Variadic<I32>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs AnyRankedTensor:$result);

  let builders = [
    // Build a ViewSliceOp with mixed static and dynamic entries and the same
    // result type
    OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = [{
    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSource().getType().getRank();
      return {rank, rank, rank};
    }
  }];

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let hasVerifier = 1;
}

#endif
