#ifndef CPP_EXAMPLE_KERNELS_KERNELINITCONTEXT_H
#define CPP_EXAMPLE_KERNELS_KERNELINITCONTEXT_H

#include <utility>

#include "NdArray.h"
#include "imaging/Metadata.h"

namespace imaging {
class KernelConstructionContext {
public:
    KernelConstructionContext(NdArrayDef input, NdArrayDef output, Metadata inputMetadata)
        : input(std::move(input)), output(std::move(output)), inputMetadata(std::move(inputMetadata)) {

        // Start with the input metadata.
        outputMetadataBuilder = MetadataBuilder{inputMetadata};
    }

    const NdArrayDef &getInput() const { return input; }

    const NdArrayDef &getOutput() const { return output; }

    void setOutput(const NdArrayDef &output) { KernelConstructionContext::output = output; }

    MetadataBuilder &getOutputMetadataBuilder() { return outputMetadataBuilder; }

private:
    NdArrayDef input, output;
    Metadata inputMetadata;
    MetadataBuilder outputMetadataBuilder;
};
}// namespace imaging
#endif//CPP_EXAMPLE_KERNELS_KERNELINITCONTEXT_H
