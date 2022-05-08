#ifndef CPP_EXAMPLE_IMAGING_OPS_ENVELOPEDETECTION_H
#define CPP_EXAMPLE_IMAGING_OPS_ENVELOPEDETECTION_H

#include "imaging/Operation.h"

namespace imaging {

/**
 * Beamforms input RF data to a sequence of high-resolution images
 * (a sum of low-resolution images).
 * The output images will reconstructed on a given grid of pixels.
 *
 * Currently works only with plane wave data.
 *
 */
class EnvelopeDetection {
public:
    EnvelopeDetection() { op = OperationBuilder{}.setClassId(OPERATION_CLASS_ID(EnvelopeDetection)).build(); }

    operator Operation() { return op; }

private:
    Operation op;
};

}



#endif//CPP_EXAMPLE_IMAGING_OPS_ENVELOPEDETECTION_H
