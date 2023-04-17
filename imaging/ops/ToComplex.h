#ifndef CPP_EXAMPLE_IMAGING_OPS_TO_COMPLEX_H
#define CPP_EXAMPLE_IMAGING_OPS_TO_COMPLEX_H

#include "imaging/Operation.h"

namespace arrus_example_imaging {

/**
 * Converts each I/Q point to its complex representation (i.e. i + j*q).
 */
class ToComplex {
public:
    ToComplex() { op = OperationBuilder{}.setClassId(OPERATION_CLASS_ID(ToComplex)).build(); }

    operator Operation() { return op; }

private:
    Operation op;
};

}



#endif//CPP_EXAMPLE_IMAGING_OPS_TO_COMPLEX_H
