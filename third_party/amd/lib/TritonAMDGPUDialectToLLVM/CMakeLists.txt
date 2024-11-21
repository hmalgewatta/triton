add_triton_library(TritonAMDGPUDialectToLLVM
    TritonAMDGPUToLLVMPatterns.cpp
    DotSlicing.cpp
    DotSlicingReordering.cpp
    ExtractSliceOpToLLVM.cpp

    DEPENDS
    TritonAMDGPUIR
)
