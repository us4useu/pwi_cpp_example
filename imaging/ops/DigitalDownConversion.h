#ifndef CPP_EXAMPLE_IMAGING_OPS_DIGITALDOWNCONVERSION_H
#define CPP_EXAMPLE_IMAGING_OPS_DIGITALDOWNCONVERSION_H

#include "imaging/Operation.h"

namespace imaging {

/**
 * Digital down conversion: demodulate, decimate by a given factor.
 */
class DigitalDownConversion {
public:
    DigitalDownConversion() { op = OperationBuilder{}.setClassId(OPERATION_CLASS_ID(DigitalDownConversion)).build(); }

    operator Operation() { return op; }

private:
    Operation op;
};

}

#endif//CPP_EXAMPLE_IMAGING_OPS_DIGITALDOWNCONVERSION_H
