add_triton_library(TritonAMDGPUDialectToLLVM
    TritonAMDGPUToLLVMPatterns.cpp
    ViewSliceOpToLLVM.cpp
    DotSlicing.cpp

    DEPENDS
    TritonAMDGPUIR
)
