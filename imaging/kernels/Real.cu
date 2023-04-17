#ifndef CPP_EXAMPLE_KERNELS_REAL_CUH
#define CPP_EXAMPLE_KERNELS_REAL_CUH

#include "Real.h"

namespace arrus_example_imaging {
__global__ void gpuReal(float *output, const float2 *input, const unsigned totalNSamples) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= totalNSamples) {
        return;
    }
    output[idx] = input[idx].x;
}

void RealFunctor::operator()(NdArray &output, const NdArray &input, cudaStream_t stream) {
    dim3 block(512);
    unsigned totalNSamples = output.getNumberOfElements();
    dim3 grid((totalNSamples+block.x-1) / block.x);
    gpuReal<<<grid, block, 0, stream>>>(output.getPtr<float>(), input.getConstPtr<float2>(), totalNSamples);
    CUDA_ASSERT(cudaGetLastError());
}
}// namespace arrus_example_imaging

#endif
