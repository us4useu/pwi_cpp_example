#ifndef CPP_EXAMPLE_NDARRAY_H
#define CPP_EXAMPLE_NDARRAY_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include "CudaUtils.cu"
#include "DataType.h"

namespace imaging {

typedef std::vector<unsigned> DataShape;
typedef DataType DataType;


class NdArrayDef {
public:
    NdArrayDef() = default;

    NdArrayDef(DataShape Shape, DataType Type) : shape(std::move(Shape)), type(Type) {}

    const DataShape &getShape() const { return shape; }
    DataType getType() const { return type; }

private:
    DataShape shape;
    DataType type;
};

class NdArray {
public:
    NdArray() = default;

    NdArray(const NdArrayDef &definition, bool isGpu)
        : ptr(nullptr), shape(definition.getShape()), dataType(definition.getType()), isGpu(isGpu) {
        if (shape.empty()) {
            // empty array shape (0)
            return;
        }
        nBytes = calculateSize(shape, dataType);
        if (isGpu) {
            CUDA_ASSERT(cudaMalloc(&ptr, nBytes));
        } else {
            CUDA_ASSERT(cudaMallocHost(&ptr, nBytes));
        }
    }

    NdArray(void *ptr, const NdArrayDef& definition, bool isGpu)
        : ptr((uint8_t *) ptr), shape(definition.getShape()), dataType(definition.getType()), isGpu(isGpu) {
        nBytes = calculateSize(shape, dataType);
        isExternal = true;
    }

    NdArray(NdArray &&array) noexcept
        : ptr(array.ptr), shape(std::move(array.shape)), dataType(array.dataType), isGpu(array.isGpu),
          nBytes(array.nBytes), isExternal(array.isExternal) {
        array.ptr = nullptr;
        array.nBytes = 0;
    }

    NdArray &operator=(NdArray &&array) noexcept {
        if (this != &array) {
            freeMemory();

            ptr = array.ptr;
            array.ptr = nullptr;

            nBytes = array.nBytes;
            array.nBytes = 0;

            shape = std::move(array.shape);
            dataType = array.dataType;
            isGpu = array.isGpu;
            isExternal = array.isExternal;
        }
        return *this;
    }

    NdArray(const NdArray &array) {
        // TODO implement
    }

    NdArray &operator=(const NdArray &input) {
        // TODO
        return *this;
    }

    virtual ~NdArray() { freeMemory(); }

    template<typename T> T *getPtr() { return (T *) ptr; }

    template<typename T> const T *getConstPtr() const { return (T *) ptr; }

    const std::vector<unsigned> &getShape() const { return shape; }

    DataType getDataType() const { return dataType; }

    size_t getNBytes() const { return nBytes; }

    bool IsGpu() const { return isGpu; }

    void freeMemory() {
        if (ptr == nullptr) {
            return;
        }
        if(isExternal) {
            // external data (views) are not managed by this class
            return;
        }
        if (isGpu) {
            CUDA_ASSERT_NO_THROW(cudaFree(ptr));
        } else {
            CUDA_ASSERT_NO_THROW(cudaFreeHost(ptr));
        }
    }

    NdArray createView() {
        return NdArray{ptr, NdArrayDef{shape, dataType}, isGpu};
    }

    bool isView() const {
        return isExternal;
    }

private:
    static size_t getSizeofDataType(DataType type) {
        if (type == DataType::UINT8) {
            return sizeof(unsigned char);
        } else if (type == DataType::INT8) {
            return sizeof(char);
        } else if (type == DataType::UINT16) {
            return sizeof(unsigned short);
        } else if (type == DataType::INT16) {
            return sizeof(short);
        } if (type == DataType::UINT32) {
            return sizeof(unsigned int);
        } else if (type == DataType::INT32) {
            return sizeof(int);
        } else if (type == DataType::FLOAT32) {
            return sizeof(float);
        } else if (type == DataType::FLOAT64) {
            return sizeof(double);
        } else if (type == DataType::COMPLEX64) {
            return sizeof(float) * 2;
        } else if (type == DataType::COMPLEX128) {
            return sizeof(double) * 2;
        }
        throw std::runtime_error("Unhandled data type");
    }

    static size_t calculateSize(const DataShape &shape, DataType type) {
        size_t result = 1;
        for (auto &val : shape) {
            result *= val;
        }
        return result * getSizeofDataType(type);
    }

    uint8_t *ptr{nullptr};
    DataShape shape{};
    size_t nBytes{0};
    DataType dataType{DataType::UINT8};
    bool isGpu{false};
    bool isExternal{false};
};
}// namespace imaging

#endif//CPP_EXAMPLE_NDARRAY_H
