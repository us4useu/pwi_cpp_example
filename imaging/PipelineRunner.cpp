#include <cmath>
#include <fstream>
#include <iostream>

#include "Metadata.h"
#include "NdArray.h"
#include "PipelineRunner.h"
#include "pwi.h"
#include "imaging/KernelRegistry.h"

namespace imaging {

PipelineRunner::PipelineRunner(std::shared_ptr<::arrus::framework::Buffer> inputBuffer, const PwiSequence &pwiSequence,
                               std::shared_ptr<::arrus::session::UploadConstMetadata> metadata,
                               Pipeline pipeline)
    : inputBuffer(std::move(inputBuffer)), metadata(std::move(metadata)), pipeline(std::move(pipeline)) {
    // Determine input shape and input data type based on
    if (inputBuffer->getNumberOfElements() == 0) {
        throw std::runtime_error("The input buffer cannot be empty");
    }
    auto element = inputBuffer->getElement(0);
    auto inputShape = element->getData().getShape();
    // FIXME int16 specific
    inputDef = NdArrayDef{inputShape.getValues(), DataType::INT16};
    inputGpu = NdArray{inputDef, true};
    inputMetadata = convertToImagingMetadata(metadata, pwiSequence);
    CUDA_ASSERT(cudaStreamCreate(&processingStream));
    prepare();
}

PipelineRunner::~PipelineRunner() { CUDA_ASSERT_NO_THROW(cudaStreamDestroy(processingStream)); }

void PipelineRunner::prepare() {

    KernelRegistry registry;
    NdArrayDef currentInputDef = inputDef;
    Metadata currentMetadata = inputMetadata;
    NdArray currentInputArray = inputGpu.createView();


    for (auto &op: pipeline.getOps()) {
        // Create Construction context: get the current NdArrayDef (start with inputDef),
        KernelConstructionContext constructionContext{currentInputDef, currentInputDef, inputMetadata};
        // determine kernel, run factory function from Registry
        // Create output array for that kernel, create context for that
        Kernel::Handle kernel = registry.createKernel(op, constructionContext);
        kernels.push_back(std::move(kernel));
        kernelOutputs.emplace_back(constructionContext.getOutput(), true);
        auto &outputArray = kernelOutputs[kernelOutputs.size()-1];
        kernelExecutionCtx.emplace_back(currentInputArray, outputArray.createView(), processingStream);
        currentInputDef = constructionContext.getOutput();
        currentMetadata = constructionContext.getOutputMetadataBuilder().build();
        currentInputArray = outputArray.createView();
    }
    this->outputDef = currentInputDef;
    outputHost = NdArray(this->outputDef, false);
}

void PipelineRunner::process(const ::arrus::framework::BufferElement::SharedHandle &ptr,
                             void (*processingCallback)(void *)) {
    // NOTE: data transfers H2D, D2H and processing are intentionally
    // serialized here into a single 'processingStream', for the sake
    // of simplicity.
    // Normally, n-element buffers should probably be used (with some
    // additional synchronization or overwrite detection) as a common
    // memory area for communication between RF data producer and
    // consumer.

    auto &inputArray = ptr->getData();
    // Wrap pointer to the input data into NdArray object.
    NdArray inputHost{inputArray.get<int16_t>(), inputDef, false};
    //    Transfer data H2D.
    CUDA_ASSERT(cudaMemcpyAsync(inputGpu.getPtr<void>(), inputHost.getPtr<void>(), inputHost.getNBytes(),
                                cudaMemcpyHostToDevice, processingStream));
    // Release host RF buffer element after transferring the data.
    CUDA_ASSERT(cudaLaunchHostFunc(
        processingStream, [](void *element) { ((::arrus::framework::BufferElement *) element)->release(); },
        ptr.get()));

    // Execute a sequence of pipeline kernels.
    for (size_t i = 0; i < kernels.size(); ++i) {
        kernels[i]->process(kernelExecutionCtx[i]);
    }
    auto &pipelineOutput = kernelOutputs[kernelOutputs.size() - 1];
    //    Transfer data D2H.
    CUDA_ASSERT(cudaMemcpyAsync(outputHost.getPtr<void>(), pipelineOutput.getPtr<void>(), outputHost.getNBytes(),
                                cudaMemcpyDeviceToHost, processingStream));
    CUDA_ASSERT(cudaStreamSynchronize(processingStream));
    processingCallback(outputHost.getPtr<void>());
    //    There seems to be some issues when calling opencv::imshow in cuda callback,
    //        so I had to use cudaStreamSynchronize here.
    //    CUDA_ASSERT(cudaLaunchHostFunc(processingStream, processingCallback, outputHost.getPtr<void>()));
}

Metadata
PipelineRunner::convertToImagingMetadata(const std::shared_ptr<::arrus::session::UploadConstMetadata> &metadata,
                                         const PwiSequence &sequence) {
    MetadataBuilder builder;
    builder.addObject("frameChannelMapping",
                      metadata->get<::arrus::devices::FrameChannelMapping>("frameChannelMapping"));
    builder.addObject("sequence", std::make_shared<PwiSequence>(sequence));
    return builder.build();
}

}// namespace imaging
