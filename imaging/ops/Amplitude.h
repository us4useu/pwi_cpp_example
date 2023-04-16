#ifndef CPP_EXAMPLE_IMAGING_OPS_AMPLITUDE_H
#define CPP_EXAMPLE_IMAGING_OPS_AMPLITUDE_H

#include "imaging/Operation.h"

namespace arrus_example_imaging {

/**
 * Converts each I/Q point to its amplitude (i.e. hypotf(i, q)).
 */
class Amplitude {
public:
    Amplitude() { op = OperationBuilder{}.setClassId(OPERATION_CLASS_ID(Amplitude)).build(); }

    operator Operation() { return op; }

private:
    Operation op;
};

}



#endif//CPP_EXAMPLE_IMAGING_OPS_AMPLITUDE_H
