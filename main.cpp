#include <iostream>
#include <thread>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "arrus/core/api/arrus.h"

using namespace ::cv;
using namespace ::arrus::session;
using namespace ::arrus::devices;
using namespace ::arrus::ops::us4r;
using namespace ::arrus::framework;

#include "constants.h"
#include "common.h"
// My custom logger, which I register in arrus.
#include "logging/MyCustomLoggerFactory.h"
#include "imaging/Pipeline.h"
#include "gui.h"
#include "menu.h"
#include "pwi.h"

// An object representing window that displays the data.
Display2D mainDisplay;

std::shared_ptr<::imaging::Pipeline> createImagingPipeline(Session *session, const PwiSequence &seq,
                                                           const std::shared_ptr<UploadConstMetadata>& metadata);

void initializeDisplay(const std::vector<unsigned int> &inputShape, imaging::DataType type);


int main() noexcept {
    try {
        // The below line register a custom logger in arrus package.
        // In order to get output for log messages with level < INFO, it is
        // necessary to register a custom logger factory. Please refer to the
        // MyCustomLoggerFactory implementation for more details.
        //
        // Also, please remember that ARRUS package only reports errors
        // by throwing exceptions, so it is therefore recommended to wrap
        // that uses ARRUS into try ..catch clauses.
        ::arrus::setLoggerFactory(std::make_shared<MyCustomLoggerFactory>(::arrus::LogSeverity::INFO));

        auto session = ::arrus::session::createSession("C:/Users/Public/us4r.prototxt");
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");

        // Creating TX/RX sequence to be executed by the device.
        constexpr unsigned N_ANGLES = 4;
        // TX angles range
        constexpr float MIN_ANGLE = -10.0f; // [deg]
        constexpr float MAX_ANGLE = 10.0f; // [deg]
        auto txAngles = linspace(MIN_ANGLE*PI/180, MAX_ANGLE*PI/180, N_ANGLES);
        PwiSequence seq {
            txAngles, // a list of transmit angles [rad]
            arrus::ops::us4r::Pulse(
                6e6f,  // center frequency [Hz]
                2,    // number of sine wave periods
                false // inverse?
            ),
            1540,     // speed of sound (assumed) [m/s]
            120e-6,   // pulse repetition interval [s]
            // Below is the time between consecutive sequence executions ("sequence repetition interval").
            // If the total PRI for a given sequence is smaller than SRI - the last TX/RX
            // pri will be increased by SRI-sum(PRI)
            50e-3,    // sequence repetition interval (an inverse of the actual b-mode frame rate) [s]
            {0, 2048},// sample range (start sample, end sample)
        };

        auto result = upload(session.get(), seq);
        // Get upload results:
        // - RF buffer, which will be filled by Us4OEMS after the session is started.
        auto buffer = std::static_pointer_cast<DataBuffer>(result.getBuffer());
        // - RF data description - currently contains only information about frame channel mapping.
        auto imgPipeline = createImagingPipeline(session.get(), seq, result.getConstMetadata());
        // Set dimensions of the window with B-mode image.
        initializeDisplay(imgPipeline->getOutputShape(), imgPipeline->getOutputDataType());
        // Register processing pipeline for RF channel data buffer.
        OnNewDataCallback callback = [&, imgPipeline, i = 0](const BufferElement::SharedHandle &ptr) mutable {
            try {
                auto* dataPtr = ptr->getData().get<int16_t>();
                // Here we do some of our custom processing.
                imgPipeline->process(
                    // Given pointer to the data.
                    dataPtr,
                    // A callback function that will be called when processing ends.
                    // In this case, just update Display.
                    [](void* input) {mainDisplay.update(input);},
                    // A callback function, that will called when a given buffer element is ready to be released.
                    // In this case, we release host buffer memory for new data.
                    [](void *element) {((BufferElement*)element)->release();},  ptr.get());
            } catch (const std::exception &e) {
                std::cout << "Exception: " << e.what() << std::endl;
            } catch (...) {
                std::cout << "Unrecognized exception" << std::endl;
            }
        };
        buffer->registerOnNewDataCallback(callback);

        // Start TX/RX sequence.
        session->startScheme();
        runMainMenu(us4r, seq);
        // Exit after stopping the scheme.
        session->stopScheme();

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}

std::shared_ptr<::imaging::Pipeline> createImagingPipeline(Session *session, const PwiSequence &seq,
                                                           const std::shared_ptr<UploadConstMetadata>& metadata) {
    // --- Copy frame channel mapping to two separate std::vectors
    auto fcm = metadata->get<FrameChannelMapping>("frameChannelMapping");
    auto nChannels = fcm->getNumberOfLogicalChannels();
    auto nFrames = fcm->getNumberOfLogicalFrames();

    // Actually, the below arrays represent 2D arrays with shape: (nFrames, nChannels).
    std::vector<int8_t> fcmChannels(nChannels*nFrames, -1);
    std::vector<uint16_t> fcmFrames(nChannels*nFrames, 0);

    // Iterate over logical frames and channels.
    for(uint16_t fr = 0; fr < nFrames; ++fr) {
        for(uint16_t ch = 0; ch < nChannels; ++ch) {
            auto [physicalFrame, physicalChannel] = fcm->getLogical(fr, ch);
            fcmFrames[fr*nChannels+ch] = physicalFrame;
            fcmChannels[fr*nChannels+ch] = physicalChannel;
        }
    }

    // -- Determine number of samples.
    auto nSamples = seq.getSampleRange().second-seq.getSampleRange().first;

    // -- Determine number of frames per TX/RX.
    constexpr unsigned N_US4OEMS = 2;
    constexpr unsigned US4OEM_N_RX = 32;

    auto probeModel = ((Us4R*)session->getDevice("/Us4R:0"))->getProbe(0)->getModel();
    unsigned nElements = probeModel.getNumberOfElements()[0];

    // Here we use all probe elements for RX aperture.
    unsigned apertureSize = nElements;
    // Number of Us4OEM output frames per each single probe's TX/RX (done by muxing RX channels).
    // Example: rx aperture has with 192 elements, we have only 32 channels per us4OEM.
    // Thus we get here 6 frames: 3 TX/RXs on the first Us4OEM module, 3 TX/RXs on the second one.
    unsigned nFramesPerAngle = apertureSize / US4OEM_N_RX;

    auto [sampleStart, sampleEnd] = seq.getSampleRange();
    unsigned nTx = seq.getAngles().size();

    // Create imaging pipeline, that will be used to reconstruct B-mode data
    return ::imaging::createPwiImagingPipeline(
        {(nTx * nFramesPerAngle) * nSamples, 32},
        fcmChannels.data(), fcmFrames.data(),
        nTx, seq.getAngles(),
        nElements, nSamples, sampleStart,
        probeModel.getPitch()[0], SAMPLING_FREQUENCY/seq.getDownsamplingFactor(),
        seq.getPulse().getCenterFrequency(),seq.getPulse().getNPeriods(),
        seq.getSpeedOfSound());
}

void initializeDisplay(const std::vector<unsigned int> &inputShape, imaging::DataType type) {
    if(inputShape.size() < 2) {
        throw std::runtime_error("Pipeline's output shape should have at least 2 dimensions.");
    }
    mainDisplay.setNrows(inputShape[inputShape.size()-2]);
    mainDisplay.setNcols(inputShape[inputShape.size()-1]);
    mainDisplay.setInputDataType(type);

}