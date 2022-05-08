#include <iostream>
#include <thread>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "arrus/core/api/arrus.h"
#include "constants.h"
#include "common.h"
// My custom logger, which I register in arrus.
#include "logging/MyCustomLoggerFactory.h"
#include "imaging/PipelineRunner.h"
#include "imaging/ops/Pipeline.h"
#include "imaging/ops/RemapToLogicalOrder.h"
#include "imaging/ProbeModelExt.h"
#include "gui.h"
#include "menu.h"
#include "pwi.h"

using namespace ::cv;
using namespace ::arrus::session;
using namespace ::arrus::devices;
using namespace ::arrus::ops::us4r;
using namespace ::arrus::framework;
using namespace ::imaging;

// An object representing window that displays the data.
Display2D mainDisplay;

//void initializeDisplay(const std::vector<unsigned int> &inputShape, imaging::DataType type);

// grid OX coordinates
constexpr float X_PIX_L = -19.0e-3f, X_PIX_R = 19.0e-3f, X_PIX_STEP = 0.1e-3;
// grid OZ coordinates
constexpr float Z_PIX_L = 5e-3f, Z_PIX_R = 42.5e-3f, Z_PIX_STEP = 0.1e-3;

// Hanning window, band (0.5, 1.5)*TX_FREQUENCY, order 64
const std::vector<float> BANDPASS_FILTER_COEFFS
    // taps for Butterworth filter, 0.5, 1.5 * 6 MHz, order 2
    {0.05892954, 0., -0.11785907, 0., 0.05892954};


// Low-pass CIC filter cooefficients.
const std::vector<float> LOWPASS_FILTER_COEFFS{1, 2, 3, 4, 3, 2, 1};

void initializeDisplay(const std::vector<unsigned int> &inputShape, imaging::DataType type) {
    if(inputShape.size() < 2) {
        throw std::runtime_error("PipelineRunner's output shape should have at least 2 dimensions.");
    }
    mainDisplay.setNrows(inputShape[inputShape.size()-2]);
    mainDisplay.setNcols(inputShape[inputShape.size()-1]);
    mainDisplay.setInputDataType(type);

}

int main() noexcept {
    try {
        // The below line register a custom logger in arrus package.
        // In order to getArray output for log messages with level < INFO, it is
        // necessary to register a custom logger factory. Please refer to the
        // MyCustomLoggerFactory implementation for more details.
        //
        // Also, please remember that ARRUS package only reports errors
        // by throwing exceptions, so it is therefore recommended to wrap
        // that uses ARRUS into try ..catch clauses.
        ::arrus::setLoggerFactory(std::make_shared<MyCustomLoggerFactory>(::arrus::LogSeverity::INFO));

        auto session = ::arrus::session::createSession("/home/pjarosik/us4r.prototxt");
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");

        auto fullArray = us4r->getProbe(0)->getModel();
        ProbeModelExt columnArray{0, fullArray, 0, 128, 0.245e-3f, std::numeric_limits<float>::infinity(), ProbeModelExt::Axis::OX};
//        ProbeModelExt rowArray{1, fullArray, 128, 256, std::numeric_limits<float>::infinity(), ProbeModelExt::Axis::OY};
        std::vector<ProbeModelExt> arrayModels = {columnArray, };// rowArray};

        std::vector<PwiSequence::Aperture> txApertures {
            columnArray.getFullAperture(),
//            columnArray.getFullAperture(),
//            columnArray.getFullAperture()
//            columnArray.getFullAperture()
        };
        std::vector<PwiSequence::Aperture> rxApertures {
            columnArray.getFullAperture(),
//            columnArray.getFullAperture(),
//            columnArray.getFullAperture()
//            columnArray.getFullAperture()
        };
        std::vector<float> txAngles = {
            0, -10, 10 // RR
//            0, // CC
        }; // [deg]

        for(size_t i = 0; i < txAngles.size(); ++i) {
            txAngles[i] = txAngles[i]*PI/180.0f;
        }
        // Now txAngles are in radians

        PwiSequence seq {
            txApertures,
            rxApertures,
            txAngles, // a list of transmit angles [rad]
            arrus::ops::us4r::Pulse(
                6e6f,  // center frequency [Hz]
                2,    // number of sine wave periods
                false // inverse?
            ),
            1540,     // speed of sound (assumed) [m/s]
            1000e-6,   // pulse repetition interval [s]
            // Below is the time between consecutive sequence executions ("sequence repetition interval").
            // If the total PRI for a given sequence is smaller than SRI - the last TX/RX
            // pri will be increased by SRI-sum(PRI)
            50e-3,    // sequence repetition interval (an inverse of the actual b-mode frame rate) [s]
            {0, 512},// sample range (start sample, end sample)
        };

        auto result = upload(session.get(), seq, arrayModels);
        // Get upload results:
        // - RF buffer, which will be filled by Us4OEMS after the session is started.
        auto buffer = std::static_pointer_cast<DataBuffer>(std::get<0>(result));
        NdArrayDef outputDef = std::get<1>(result);
        auto metadata = std::get<2>(result);

        // - RF data description - currently contains only information about frame channel mapping.

        PipelineRunner runner {
            outputDef,
            metadata,
            // Processing steps to be performed on GPU.
            Pipeline{{
                RemapToLogicalOrder{},
            }}
        };

        // Set dimensions of the window with B-mode image.
        initializeDisplay(runner.getOutputDef().getShape(), runner.getOutputDef().getType());
        // Register processing pipeline for RF channel data buffer.
        OnNewDataCallback callback = [&runner](const BufferElement::SharedHandle &ptr) mutable {
            try {
                // Here we do some of our custom processing.
                runner.process(
                    // Given pointer to the data.
                    ptr,
                    // A callback function that will be called when processing ends.
                    // In this case, just update Display.
                    [](void* input) {mainDisplay.update(input);}
                );
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
        std::cout << "Done" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

