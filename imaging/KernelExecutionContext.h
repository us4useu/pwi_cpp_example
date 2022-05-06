#ifndef PWI_CPP_EXAMPLE_IMAGING_EXECUTIONCONTEXT_H
#define PWI_CPP_EXAMPLE_IMAGING_EXECUTIONCONTEXT_H

namespace imaging {

class KernelExecutionContext {
public:
    KernelExecutionContext(const NdArray &input, const NdArray &output, const cudaStream_t stream)
        : input(input), output(output), stream(stream) {}

    const NdArray &getInput() const { return input; }
    const NdArray &getOutput() const { return output; }
    const cudaStream_t getStream() const { return stream; }

private:
    // View
    NdArray input;
    // View
    NdArray output;
    cudaStream_t stream;
};

}

#endif //PWI_CPP_EXAMPLE_IMAGING_EXECUTIONCONTEXT_H
