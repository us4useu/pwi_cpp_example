#ifndef CPP_EXAMPLE_IMAGING_OPERATION_H
#define CPP_EXAMPLE_IMAGING_OPERATION_H

#include <string>
#include <utility>
#include <unordered_map>
#include "NdArray.h"

namespace imaging {

#define OPERATION_CLASS_ID(Type) #Type

class OpParameters {
public:
    using Container = std::unordered_map<std::string, NdArray>;

    OpParameters() {}

    explicit OpParameters(Container params)
        : params(std::move(params)) {}

    NdArray& get(const std::string &key) {
        return params.at(key);
    }
private:
    Container params;
};

class Operation {
public:
    using OpClassId = std::string;
    using OpId = std::string;

    Operation() {}

    Operation(OpClassId classId, const OpParameters &params)
        : classId(std::move(classId)), params(params) {}

    OpClassId getOpClassId() const { return classId; }

private:
    OpClassId classId;
    OpParameters params;
};

class OperationBuilder {

public:
    OperationBuilder() = default;

    OperationBuilder& setClassId(Operation::OpClassId id) {
        this->classId = std::move(id);
        return *this;
    }

    OperationBuilder& addParam(const std::string& key, const NdArray arr) {
//        this->params.insert(key, std::move(arr));
        return *this;
    }

    Operation build() {
        return Operation{classId, OpParameters{params}};
    }

private:
    Operation::OpClassId classId;
    OpParameters::Container params;
};

}// namespace imaging

#endif//CPP_EXAMPLE_IMAGING_OPERATION_H
