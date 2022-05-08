#include "../NdArray.h"
#include "../KernelInitResult.h"
#include "../KernelConstructionContext.h"
#include "../Kernel.h"

namespace imaging {

__global__ void gpuDecimation(float2 *output, const float2 *input,
                              const unsigned nSamples,
                              const unsigned totalNSamples,
                              const unsigned decimationFactor) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= totalNSamples) {
        return;
    }
    int decimatedNSamples = (int) ceilf((float) nSamples / decimationFactor);
    output[idx] = input[
        (idx / decimatedNSamples) * nSamples // (transmit, channel number)
        + (idx % decimatedNSamples) * decimationFactor];
}

class Decimation : public Kernel {
public:

    KernelInitResult prepare(const KernelInitContext &ctx) override {
        auto &inputShape = ctx.getInputShape();
        auto inputDtype = ctx.getDataType();

        if (inputShape.size() != 3) {
            throw std::runtime_error(
                "Currently decimation works only with 3D arrays");
        }
        this->nSamples = inputShape[2];
        auto outputNSamples = (unsigned) ceilf(
            (float) nSamples / decimationFactor);
        this->outputTotalNSamples =
            inputShape[0] * inputShape[1] * outputNSamples;
        return KernelInitResult(
            {inputShape[0], inputShape[1], outputNSamples},
            inputDtype,
            ctx.getInputSamplingFrequency() / decimationFactor);
    }

    void process(NdArray *output,
                 const NdArray *input,
                 cudaStream_t &stream) override {
        dim3 block(512);
        dim3 grid((outputTotalNSamples + block.x - 1) / block.x);
        gpuDecimation <<<grid, block, 0, stream >>>(
            output->getPtr<float2>(), input->getConstPtr<float2>(),
            nSamples,
            outputTotalNSamples,
            decimationFactor);
        CUDA_ASSERT(cudaGetLastError());
    }

    Decimation(unsigned decimationFactor) : decimationFactor(
        decimationFactor) {}

private:
    unsigned outputTotalNSamples{0};
    unsigned nSamples{0};
    unsigned decimationFactor;
};
}

#include "../Kernel.h"

namespace imaging {

#define MAX_CIC_SIZE 512

__device__ __constant__ float gpuCicCoefficients[MAX_CIC_SIZE];

__global__ void gpuFirLp(
    float2 *__restrict__ output, const float2 *__restrict__ input,
    const int nSamples, const int totalNSamples, const int kernelWidth) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ch = idx / nSamples;
    int sample = idx % nSamples;

    extern __shared__ char sharedMemory[];

    float2 *cachedInputData = (float2 *) sharedMemory;
    // Cached input data stores all the input data which is convolved with given
    // filter.
    // That means, there should be enough input data from the last thread in
    // the thread group to compute convolution.
    // Thus the below condition localIdx < (blockDim.x + kernelWidth)
    // Cache input.
    for (int i = sample - kernelWidth / 2 - 1, localIdx = threadIdx.x;
         localIdx <
         (kernelWidth + blockDim.x); i += blockDim.x, localIdx += blockDim.x) {
        if (i < 0 || i >= nSamples) {
            cachedInputData[localIdx] = make_float2(0.0f, 0.0f);
        } else {
            cachedInputData[localIdx] = input[ch * nSamples + i];
        }
    }
    __syncthreads();
    if (idx >= totalNSamples) {
        return;
    }
    float2 result = make_float2(0.0f, 0.0f);

    int localN = threadIdx.x + kernelWidth;
    for (int i = 0; i < kernelWidth; ++i) {
        result.x += cachedInputData[localN - i].x * gpuCicCoefficients[i];
        result.y += cachedInputData[localN - i].y * gpuCicCoefficients[i];
    }
    output[idx] = result;
}

/**
 * FIR filter. NOTE: there should be only one instance of this kernel in a single
 * imaging pipeline.
 * The constraint on the number of instances is due to the usage of
 * global constant memory to store filter coefficients.
 */
class LpFilterSingleton : public Kernel {
public:

    explicit LpFilterSingleton(const std::vector<float> coefficients) {
        this->nCoefficients = coefficients.size();
        CUDA_ASSERT(cudaMemcpyToSymbol(
            gpuCicCoefficients,
            coefficients.data(),
            coefficients.size() * sizeof(float),
            0, cudaMemcpyHostToDevice));
    }

    KernelInitResult prepare(const KernelInitContext &ctx) override {
        auto &inputShape = ctx.getInputShape();
        auto inputDtype = ctx.getDataType();

        if (inputShape.size() != 3) {
            throw std::runtime_error(
                "Currently fir filter works only with 3D arrays");
        }
        this->totalNSamples = inputShape[0] * inputShape[1] * inputShape[2];
        this->nSamples = inputShape[2];
        return KernelInitResult(
            inputShape, inputDtype, ctx.getInputSamplingFrequency());
    }

    void process(NdArray *output, const NdArray *input,
                 cudaStream_t &stream) override {
        dim3 filterBlockDim(512);
        dim3 filterGridDim(
            (this->totalNSamples + filterBlockDim.x - 1) /
            filterBlockDim.x);
        unsigned sharedMemSize =
            (filterBlockDim.x + nCoefficients) * sizeof(float2);
        gpuFirLp<<<filterGridDim, filterBlockDim, sharedMemSize, stream >>>(
            output->getPtr<float2>(), input->getConstPtr<float2>(),
            this->nSamples, this->totalNSamples, this->nCoefficients);
        CUDA_ASSERT(cudaGetLastError());
    }

private:
    unsigned totalNSamples{0};
    unsigned nSamples{0};
    unsigned nCoefficients{0};
};

}

#include "../NdArray.h"
#include "math_constants.h"

namespace imaging {

__global__ void gpuRfToIq(float2 *output, const float *input,
                          const float sampleCoeff,
                          const unsigned nSamples,
                          const unsigned maxThreadId) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= maxThreadId) {
        return;
    }
    float rfSample = input[idx];
    int sampleNumber = idx % nSamples;
    float cosinus, sinus;
    __sincosf(sampleCoeff * sampleNumber, &sinus, &cosinus);
    float2 iq;
    iq.x = 2.0f * rfSample * cosinus;
    iq.y = 2.0f * rfSample * sinus;
    output[idx] = iq;
}

class QuadratureDemodulation : public Kernel {
public:
    QuadratureDemodulation(float transmitFrequency) :
                                                      transmitFrequency(transmitFrequency) {}

    KernelInitResult prepare(const KernelInitContext &ctx) override {
        auto &inputShape = ctx.getInputShape();
        if (inputShape.size() != 3) {
            throw std::runtime_error(
                "Currently demodulation works only with 3D arrays");
        }
        auto samplingFrequency = ctx.getSamplingFrequency();
        this->totalNSamples = inputShape[0] * inputShape[1] * inputShape[2];
        this->nSamples = inputShape[2];
        this->sampleCoeff =
            -2.0f * CUDART_PI_F * transmitFrequency / samplingFrequency;
        return KernelInitResult(
            inputShape, NdArray::DataType::COMPLEX64,
            ctx.getInputSamplingFrequency());
    }

    void process(NdArray *output, const NdArray *input,
                 cudaStream_t &stream) override {
        dim3 filterBlockDim(512);
        dim3 filterGridDim(
            (this->totalNSamples + filterBlockDim.x - 1) /
            filterBlockDim.x);
        gpuRfToIq<<<filterGridDim, filterBlockDim, 0, stream >>>(
            output->getPtr<float2>(), input->getConstPtr<float>(),
            this->sampleCoeff, this->nSamples, this->totalNSamples);
        CUDA_ASSERT(cudaGetLastError());
    }


private:
    unsigned totalNSamples{0};
    unsigned nSamples{0};
    unsigned nCoefficients{0};
    float samplingFrequency;
    float transmitFrequency;
    float sampleCoeff{0};
};

}
