#ifndef CPP_EXAMPLE_KERNELS_TO_COMPLEX_CUH
#define CPP_EXAMPLE_KERNELS_TO_COMPLEX_CUH

#include "ToComplex.h"

namespace arrus_example_imaging {
__global__ void gpuToComplex(float2 *output, const int16_t *input, const unsigned totalNSamples) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= totalNSamples) {
        return;
    }
    int16_t i = input[2*idx];
    int16_t q = input[2*idx+1];
    output[idx] = make_float2((float)i, (float)q);
}

void ToComplexFunctor::operator()(NdArray &output, const NdArray &input, cudaStream_t stream) {
    dim3 block(512);
    unsigned totalNSamples = output.getNumberOfElements();
    dim3 grid((totalNSamples+block.x-1) / block.x);
    gpuToComplex<<<grid, block, 0, stream>>>(output.getPtr<float2>(), input.getConstPtr<int16_t>(),
                                                     totalNSamples);
    CUDA_ASSERT(cudaGetLastError());
}
}// namespace arrus_example_imaging

#endif
