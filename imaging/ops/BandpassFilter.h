#ifndef CPP_EXAMPLE_IMAGING_OPS_BANDPASSFILTER_H
#define CPP_EXAMPLE_IMAGING_OPS_BANDPASSFILTER_H

#include "imaging/Operation.h"

namespace arrus::imaging {

/**
 * Applies given bandpass filter.
 */
class BandpassFilter {
public:
    explicit BandpassFilter(const NdArray &coeffs) {
        op = OperationBuilder{}
                 .setClassId(OPERATION_CLASS_ID(BandpassFilter))
                 .addParam("coefficients", coeffs)
                 .build();
    }

    operator Operation() { return op; }

private:
    Operation op;
};

}// namespace arrus::imaging
#endif//CPP_EXAMPLE_IMAGING_OPS_BANDPASSFILTER_H
