#include "EnvelopeDetection.h"

namespace imaging {

EnvelopeDetectionKernel::EnvelopeDetectionKernel(KernelConstructionContext &ctx) : Kernel(ctx) {
    auto &input = ctx.getInput();
    auto &inputShape = input.getShape();
    auto inputType = input.getType();
    ctx.setOutput(NdArrayDef{inputShape, DataType::FLOAT32});
}
void EnvelopeDetectionKernel::process(KernelExecutionContext &ctx) {
    impl(ctx.getOutput(), ctx.getInput(), ctx.getStream());
}

}