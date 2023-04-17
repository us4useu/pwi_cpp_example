#ifndef CPP_EXAMPLE_IMAGING_OPS_REAL_H
#define CPP_EXAMPLE_IMAGING_OPS_REAL_H

#include "imaging/Operation.h"

namespace arrus_example_imaging {

/**
 * Converts each I/Q point to its real value (i.e. I value).
 */
class Real {
public:
    Real() { op = OperationBuilder{}.setClassId(OPERATION_CLASS_ID(Real)).build(); }

    operator Operation() { return op; }

private:
    Operation op;
};

}



#endif//CPP_EXAMPLE_IMAGING_OPS_REAL_H
