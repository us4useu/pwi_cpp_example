#ifndef CPP_EXAMPLE_IMAGING_OPS_REMAPTOLOGICALORDER_H
#define CPP_EXAMPLE_IMAGING_OPS_REMAPTOLOGICALORDER_H

#include "imaging/Operation.h"

namespace imaging {

/**
 * Reorders input array columns according to
 * order imposed by us4R devices.
 */
class RemapToLogicalOrder {
public:
    RemapToLogicalOrder() { op = OperationBuilder{}.setClassId(OPERATION_CLASS_ID(RemapToLogicalOrder)).build(); }

    operator Operation() { return op; }

private:
    Operation op;
};

}// namespace imaging

#endif//CPP_EXAMPLE_IMAGING_OPS_REMAPTOLOGICALORDER_H
