#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
namespace tta = mlir::triton::amdgpu;

// clang-format off
/***
   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
   # WO   #  W1 #                                     |                                #
   #      #     #                                     |                                #
   #  #   #  #  #                                     |                                #
   # W2   # W3  #   ....                              |                                #
   #      #     #                                     |  SkipElems                     #
   #  #   #  #  #                                     |                                #
   #                                                  |                                #
   #                                        Slice     |                                #
   #    .                                 /        \  |                                #
   #    .                                /          \ |                                #
   #    .                               /            \|                                #
   #                                    #   #  #  #  #                                 #
   #                                    #  W0  #  W1 #                                 #
   #                                    #      #     #                                 #
   #                                    #  #   #  #  #    tensorStride                 #
   #                                    #  W2  #  W3 # --------------------------------#
   #                                    #      #     #                                 #
   #                                    #  #   #  #  #                                 #
   #          tensorStride              #  W0  #  W1 #                                 #
   # ---------------------------------- #      #     #                                 #
   #                                    #  #   #  #  #                                 #
   #                                    #  W2  #  W3 #                                 #
   #                                    #      #     #                                 #
   #                                    #  #   #  #  # ---> lastIdx                    #
   #                                         .                                         #
   #                                         .                                         #
   #                                         .                                         #
   #                                                                                   #
   #                                                                                   #
   #                                                                                   #
   #                                                                                   #
   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
***/
// clang-format on

namespace {
struct ViewSliceOpConversion : public ConvertOpToLLVMPattern<tta::ViewSliceOp> {
  explicit ViewSliceOpConversion(LLVMTypeConverter &typeConverter,
                                 PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<tta::ViewSliceOp>(typeConverter, benefit) {}

  LogicalResult processLayout(tta::ViewSliceOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto srcTy = dyn_cast<RankedTensorType>(op.getSource().getType());
    auto srcLayout = srcTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto vals = unpackLLElements(loc, adaptor.getSource(), rewriter);
    auto elemsPerThread = mlir::triton::gpu::getElemsPerThread(srcTy);
    auto sizePerThread = getSizePerThread(srcLayout);
    auto totalSizePerThread = sizePerThread[0] * sizePerThread[1];
    auto order = getOrder(srcLayout);
    auto shapePerCTA = getShapePerCTATile(srcLayout, srcShape);
    shapePerCTA[0] =
        std::min(static_cast<unsigned>(srcShape[0]), shapePerCTA[0]);
    shapePerCTA[1] =
        std::min(static_cast<unsigned>(srcShape[1]), shapePerCTA[1]);

    auto offsets = op.getStaticOffsets();
    auto sizes = op.getStaticSizes();

    // Calculate offsets and sizes in terms of CTA units.
    std::vector<long int> CTAOffsets{offsets[0] / shapePerCTA[0],
                                     offsets[1] / shapePerCTA[1]};
    std::vector<long int> CTASizes{sizes[0] / shapePerCTA[0],
                                   sizes[1] / shapePerCTA[1]};
    std::vector<long int> CTAPerShape{srcShape[0] / shapePerCTA[0],
                                      srcShape[1] / shapePerCTA[1]};

    // The diagram above illustrates the graphical representation of the
    // skipElems, tensorStride, and lastIdx variables.
    // llvm::outs() << "=========================================" << "]\n";
    // llvm::outs() << "srcShape: [" << srcShape[0] << "," << srcShape[1] << "]\n";
    // // llvm::outs() << "warpsPerCTA: [" << dyn_cast<BlockedEncodingAttr>(srcLayout).getWarpsPerCTA()[0] << "," << dyn_cast<BlockedEncodingAttr>(srcLayout).getWarpsPerCTA()[1] << "]\n";
    // llvm::outs() << "elemsPerThread: [" << elemsPerThread[0] << "," << elemsPerThread[1] << "]\n";
    // llvm::outs() << "sizePerThread: [" << sizePerThread[0] << "," << sizePerThread[1] << "]\n";
    // llvm::outs() << "shapePerCTA: [" << shapePerCTA[0] << "," << shapePerCTA[1] << "]\n";
    // llvm::outs() << "CTAOffsets: [" << CTAOffsets[0] << "," << CTAOffsets[1] << "]\n";
    // llvm::outs() << "CTASizes: [" << CTASizes[0] << "," << CTASizes[1] << "]\n";
    // llvm::outs() << "CTAPerShape: [" << CTAPerShape[0] << "," << CTAPerShape[1] << "]\n";
    auto skipElems = CTAOffsets[order[1]] *
                         (elemsPerThread[order[0]] * sizePerThread[order[1]]) +
                     CTAOffsets[order[0]] * totalSizePerThread;
    auto tensorStride =
        (CTAPerShape[order[0]] - CTASizes[order[0]]) * totalSizePerThread;
    auto lastIdx =
        (CTAOffsets[order[1]] + CTASizes[order[1]] - 1) *
            elemsPerThread[order[0]] * sizePerThread[order[1]] +
        (CTAOffsets[order[0]] + CTASizes[order[0]]) * totalSizePerThread;

    
    // llvm::outs() << "skipElems: " << skipElems << "\n";
    // llvm::outs() << "tensorStride: " << tensorStride << "\n";
    // llvm::outs() << "lastIdx: " << lastIdx << "\n";
    // llvm::outs() << "totalSizePerThread: " << totalSizePerThread << "\n";
    // llvm::outs() << "****************************************" << "]\n";
    assert(lastIdx <= vals.size());

    SmallVector<Value> resultVals;
    for (int i = skipElems; i < lastIdx; i += tensorStride) {
      for (int j = 0; j < totalSizePerThread * CTASizes[order[0]]; ++j, ++i) {
        assert(i < lastIdx);
        resultVals.push_back(vals[i]);
      }
    }
    Value ret = packLLElements(loc, this->getTypeConverter(), resultVals,
                               rewriter, resultTy);

    rewriter.replaceOp(op, ret);
    return success();
  }

  LogicalResult
  matchAndRewrite(tta::ViewSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSource().getType();
    if (isa<BlockedEncodingAttr>(op.getSource().getType().getEncoding()) ||
        isa<AMDMfmaEncodingAttr>(op.getSource().getType().getEncoding())) {
      return processLayout(op, adaptor, rewriter);
    } else {
      assert(false && "Unsupported layout in viewSlice.");
      return failure();
    }
  }
};
} // namespace

namespace mlir::triton::AMD {

void populateViewSliceOpTritonAMDGPUToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ViewSliceOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
