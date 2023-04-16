#include "Amplitude.h"

namespace arrus_example_imaging {

AmplitudeKernel::AmplitudeKernel(KernelConstructionContext &ctx) : Kernel(ctx) {
    auto &input = ctx.getInput();
    auto &inputShape = input.getShape();
    auto inputType = input.getType();  // Expected to be int16
    // Remove the "components dimension".
    DataShape outputShape;
    std::copy(std::begin(inputShape), std::end(inputShape)-1, std::back_inserter(outputShape));
    ctx.setOutput(NdArrayDef{outputShape, DataType::FLOAT32});
}
void AmplitudeKernel::process(KernelExecutionContext &ctx) {
    impl(ctx.getOutput(), ctx.getInput(), ctx.getStream());
}
REGISTER_KERNEL_OP(OPERATION_CLASS_ID(Amplitude), AmplitudeKernel)

}