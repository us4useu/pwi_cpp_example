#ifndef CPP_EXAMPLE_IMAGING_OPS_PHASE_H
#define CPP_EXAMPLE_IMAGING_OPS_PHASE_H

#include "imaging/Operation.h"

namespace arrus_example_imaging {

/**
 * Extracts I/Q phase.
 */
class Phase {
public:
    Phase() { op = OperationBuilder{}.setClassId(OPERATION_CLASS_ID(Phase)).build(); }


    operator Operation() { return op; }

private:
    Operation op;
};

}



#endif//CPP_EXAMPLE_IMAGING_OPS_PHASE_H
