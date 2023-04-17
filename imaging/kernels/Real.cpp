#include "Real.h"

namespace arrus_example_imaging {

RealKernel::RealKernel(KernelConstructionContext &ctx) : Kernel(ctx) {
    auto &input = ctx.getInput();
    auto &inputShape = input.getShape();
    auto inputType = input.getType();
    ctx.setOutput(NdArrayDef{inputShape, DataType::FLOAT32});
}
void RealKernel::process(KernelExecutionContext &ctx) {
    impl(ctx.getOutput(), ctx.getInput(), ctx.getStream());
}
REGISTER_KERNEL_OP(OPERATION_CLASS_ID(Real), RealKernel)

}