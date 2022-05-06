#ifndef CPP_EXAMPLE_PIPELINE_RUNNER_H
#define CPP_EXAMPLE_PIPELINE_H

#include <algorithm>
#include <arrus/core/api/arrus.h>
#include <functional>
#include <utility>
#include <vector>

#include "imaging/DataType.h"
#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/Pipeline.h"
#include "pwi.h"

namespace imaging {

class PipelineRunner {
public:
    explicit PipelineRunner(std::shared_ptr<::arrus::framework::Buffer> inputBuffer,
                            const PwiSequence &sequence,
                            std::shared_ptr<::arrus::session::UploadConstMetadata> metadata,
                            Pipeline pipeline);

    virtual ~PipelineRunner();

    void process(const ::arrus::framework::BufferElement::SharedHandle &ptr, void (*processingCallback)(void *));

private:
    void prepare();

    Metadata convertToImagingMetadata(const std::shared_ptr<::arrus::session::UploadConstMetadata> &metadata,
                                      const PwiSequence &sequence);

    std::shared_ptr<::arrus::framework::Buffer> inputBuffer;
    std::shared_ptr<::arrus::session::UploadConstMetadata> metadata;
    Pipeline pipeline;

    std::vector<Kernel::Handle> kernels;
    std::vector<KernelExecutionContext> kernelExecutionCtx;
    NdArray inputGpu;
    Metadata inputMetadata;
    // Actual arrays.
    std::vector<NdArray> kernelOutputs;
    NdArray outputHost;
    NdArrayDef inputDef, outputDef;
    cudaStream_t processingStream;

};
}// namespace imaging
#endif//CPP_EXAMPLE_PIPELINE_RUNNER_H
