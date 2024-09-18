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

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

// clang-format off
#include "Dialect/AMDGPU/IR/Dialect.h"
#include "Dialect/AMDGPU/IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::gpu;
using namespace mlir::triton::amdgpu;

void mlir::triton::amdgpu::AMDGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/AMDGPU/IR/AMDGPUAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/AMDGPU/IR/Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Dialect/AMDGPU/IR/Ops.cpp.inc"

namespace mlir::triton::amdgpu {

LogicalResult ViewSliceOp::verify() {
  auto srcTy = getSource().getType();
  auto srcLayout = srcTy.getEncoding();
  auto srcElementType = dyn_cast<RankedTensorType>(srcTy).getElementType();
  auto resultTy = getResult().getType();
  auto resultLayout = resultTy.getEncoding();
  auto resultElementType =
      dyn_cast<RankedTensorType>(resultTy).getElementType();

  if (srcElementType != resultElementType) {
    return emitError("result type must match source type");
  }

  if (srcLayout != resultLayout)
    return emitError("result layout must match source layout");

  auto elemsPerThread = mlir::triton::gpu::getElemsPerThread(srcTy);
  auto sizePerThread = mlir::triton::gpu::getSizePerThread(srcLayout);
  auto totalSizePerThread = sizePerThread[0] * sizePerThread[1];
  auto order = mlir::triton::gpu::getOrder(srcLayout);
  auto srcShape = srcTy.getShape();
  auto shapePerCTA = mlir::triton::gpu::getShapePerCTATile(srcLayout, srcShape);
  shapePerCTA[0] = std::min(srcShape[0], (long)shapePerCTA[0]);
  shapePerCTA[1] = std::min(srcShape[1], (long)shapePerCTA[1]);

  auto offsets = getStaticOffsets();
  auto sizes = getStaticSizes();

  if (offsets[0] % shapePerCTA[0] != 0 || offsets[1] % shapePerCTA[1] != 0) {
    return emitError("incorrect offset");
  }

  if (sizes[0] % shapePerCTA[0] != 0 || sizes[1] % shapePerCTA[1] != 0) {
    return emitError("incorrect size");
  }

  if (!hasUnitStride()) {
    return emitError("unsupported stride");
  }

  return success();
}
} // namespace mlir::triton::amdgpu
