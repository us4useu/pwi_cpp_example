#ifndef CPP_EXAMPLE_KERNELS_KERNELINITCONTEXT_H
#define CPP_EXAMPLE_KERNELS_KERNELINITCONTEXT_H

// TODO inputSamplingFrequency -> all other context settings

#include <utility>

#include "NdArray.h"

namespace imaging {
class KernelConstructionContext {
public:
    KernelConstructionContext(NdArray::DataShape inputShape, NdArray::DataType inputDataType,
                              float inputSamplingFrequency)
        : inputShape(std::move(inputShape)), inputDataType(inputDataType),
          inputSamplingFrequency(inputSamplingFrequency) {}

    float getInputSamplingFrequency() const { return inputSamplingFrequency; }

    const NdArray::DataShape &getInputShape() const { return inputShape; }

    NdArray::DataType getDataType() const { return inputDataType; }

    float getSamplingFrequency() const { return inputSamplingFrequency; }

    const NdArray::DataShape &getOutputShape() const { return outputShape; }

    void setOutputShape(const NdArray::DataShape &OutputShape) { outputShape = OutputShape; }

    DataType getOutputDataType() const { return outputDataType; }

    void setOutputDataType(DataType OutputDataType) { outputDataType = OutputDataType; }

    float getOutputSamplingFrequency() const { return outputSamplingFrequency; }

    void setOutputSamplingFrequency(float OutputSamplingFrequency) {
        outputSamplingFrequency = OutputSamplingFrequency;
    }

private:
    NdArray::DataShape inputShape;
    NdArray::DataType inputDataType;
    float inputSamplingFrequency;
    NdArray::DataShape outputShape;
    NdArray::DataType outputDataType;
    float outputSamplingFrequency;
};
}// namespace imaging
#endif//CPP_EXAMPLE_KERNELS_KERNELINITCONTEXT_H
