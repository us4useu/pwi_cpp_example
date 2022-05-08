#include "Transpose.h"

namespace imaging {

TransposeKernel::TransposeKernel(KernelConstructionContext &ctx) : Kernel(ctx) {
    auto &input = ctx.getInput();
    auto &inputShape = input.getShape();
    auto inputType = input.getType();
    if(inputShape.size() < 2) {
        throw std::runtime_error("The transposition can be performed on arrays with at least two arrays.");
    }
    DataShape outputShape;
    outputShape = inputShape;
    size_t rank = outputShape.size()-1;
    std::swap(outputShape[rank-1], outputShape[rank-2]);

    nColumns = outputShape[rank-1];
    nRows = outputShape[rank-2];
    nMatrices = 1;
    if(rank > 2) {
        for(int i = 0; i < rank-2; ++i) {
            nMatrices *= outputShape[i];
        }
    }
    ctx.setOutput(NdArrayDef{outputShape, inputType});
}
void TransposeKernel::process(KernelExecutionContext &ctx) {
    impl(ctx.getOutput(), ctx.getInput(), nMatrices, nRows, nColumns, ctx.getStream());
}

}
