#include <cmath>

#include "pwi.h"

using namespace ::arrus::session;
using namespace ::arrus::ops::us4r;
using namespace ::arrus::framework;

TxRxSequence createPwiSequence(const PwiSequence &seq, const arrus::devices::ProbeModel &probeModel) {
    // Apertures
    auto nElements = probeModel.getNumberOfElements()[0];
    auto apertureSize = nElements;
    std::vector<bool> rxAperture(apertureSize, true);
    std::vector<bool> txAperture(apertureSize, true);

    // Delays
    std::vector<float> delays(nElements, 0.0f);
    std::vector<TxRx> txrxs;

    float pitch = probeModel.getPitch()[0];

    for (auto angle: seq.getAngles()) {
        std::vector<float> delays(nElements, 0.0f);

        // Compute array of TX delays.
        for (int i = 0; i < nElements; ++i) {
            delays[i] = pitch * i * sin(angle) / seq.getSpeedOfSound();
        }
        float minDelay = *std::min_element(std::begin(delays), std::end(delays));
        for (int i = 0; i < nElements; ++i) {
            delays[i] -= minDelay;
        }
        txrxs.emplace_back(Tx(txAperture, delays, seq.getPulse()), Rx(rxAperture, seq.getSampleRange()),
                           seq.getPri());
    }
    float sri = seq.getSri().has_value() ? seq.getSri().value() : TxRxSequence::NO_SRI;
    return TxRxSequence{txrxs, {}, sri};
}


UploadResult upload(Session *session, const PwiSequence &seq) {
    auto *us4r =  (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");
    auto probeModel = us4r->getProbe(0)->getModel();

    auto txRxSequence = createPwiSequence(seq, probeModel);

    DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 4};
    Scheme scheme{txRxSequence, 2, outputBuffer, Scheme::WorkMode::HOST};
    return session->upload(scheme);
}

