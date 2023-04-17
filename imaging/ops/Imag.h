#ifndef CPP_EXAMPLE_IMAGING_OPS_IMAG_H
#define CPP_EXAMPLE_IMAGING_OPS_IMAG_H

#include "imaging/Operation.h"

namespace arrus_example_imaging {

/**
 * Converts each I/Q point to its imaginary part (i.e. Q value).
 */
class Imag {
public:
    Imag() { op = OperationBuilder{}.setClassId(OPERATION_CLASS_ID(Imag)).build(); }

    operator Operation() { return op; }

private:
    Operation op;
};

}



#endif//CPP_EXAMPLE_IMAGING_OPS_IMAG_H
