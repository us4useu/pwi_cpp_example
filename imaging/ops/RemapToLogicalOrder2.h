#ifndef CPP_EXAMPLE_IMAGING_OPS_REMAPTOLOGICALORDER2_H
#define CPP_EXAMPLE_IMAGING_OPS_REMAPTOLOGICALORDER2_H

#include "imaging/Operation.h"

namespace arrus_example_imaging {

/**
 * Reorders input array columns according to
 * order imposed by us4R devices. 2nd version, works with hardware demodulated data.
 */
class RemapToLogicalOrder2 {
public:
    RemapToLogicalOrder2() { op = OperationBuilder{}.setClassId(OPERATION_CLASS_ID(RemapToLogicalOrder2)).build(); }

    operator Operation() { return op; }

private:
    Operation op;
};

}// namespace arrus_example_imaging

#endif//CPP_EXAMPLE_IMAGING_OPS_REMAPTOLOGICALORDER2_H
