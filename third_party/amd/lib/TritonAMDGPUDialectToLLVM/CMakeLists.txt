add_triton_library(TritonAMDGPUDialectToLLVM
    TritonAMDGPUToLLVMPatterns.cpp
    ExtractSliceOpToLLVM.cpp
    DotSlicing.cpp
    DotSlicingReordering.cpp

    DEPENDS
    TritonAMDGPUIR
)
