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
#include "imaging/ProbeModelExt.h"

#include "imaging/ops/RemapToLogicalOrder2.h"
#include "imaging/ops/ToComplex.h"
#include "imaging/ops/EnvelopeDetection.h"
#include "imaging/ops/Real.h"

#include "gui.h"
#include "menu.h"
#include "imaging/pwi.h"

using namespace ::cv;
using namespace ::arrus::session;
using namespace ::arrus::devices;
using namespace ::arrus::ops::us4r;
using namespace ::arrus::framework;
using namespace ::arrus_example_imaging;

// An object representing window that displays the data.
Display2D mainDisplay;

// grid OX coordinates
void initializeDisplay(const std::vector<size_t> &inputShape, ::arrus_example_imaging::DataType type) {
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

        auto session = ::arrus::session::createSession("/home/pjarosik/src/x-files/customers/brainlab/us4r.prototxt");
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");

        auto fullArray = us4r->getProbe(0)->getModel();
        ProbeModelExt columnArray{0, fullArray, 0, 128, 0.245e-3f, std::numeric_limits<float>::infinity(), ProbeModelExt::Axis::OX};
//        ProbeModelExt rowArray{1, fullArray, 128, 256, std::numeric_limits<float>::infinity(), ProbeModelExt::Axis::OY};
        std::vector<ProbeModelExt> arrayModels = {columnArray, };// rowArray};

        std::vector<PwiSequence::Aperture> txApertures {
            columnArray.getFullAperture(),
            columnArray.getFullAperture(),
            columnArray.getFullAperture()
        };
        std::vector<PwiSequence::Aperture> rxApertures {
            columnArray.getFullAperture(),
            columnArray.getFullAperture(),
            columnArray.getFullAperture()
        };
        std::vector<float> txAngles = {
            0, -10, 10 // CC
        }; // [deg]
        for(size_t i = 0; i < txAngles.size(); ++i) {
            txAngles[i] = txAngles[i]*PI/180.0f;
        }
        // Now txAngles are in radians

        float txFrequency = 6e6f; // [Hz]
        PwiSequence seq {
            txApertures,
            rxApertures,
            txAngles, // a list of transmit angles [rad]
            arrus::ops::us4r::Pulse(
                txFrequency,  // center frequency [Hz]
                2,    // number of sine wave periods
                false // inverse?
            ),
            1540,     // speed of sound (assumed) [m/s]
            1000e-6,   // pulse repetition interval [s]
            // Below is the time between consecutive sequence executions ("sequence repetition interval").
            // If the total PRI for a given sequence is smaller than SRI - the last TX/RX
            // pri will be increased by SRI-sum(PRI)
            50e-3,    // sequence repetition interval (an inverse of the actual b-mode frame rate) [s]
            {0, 1024},// sample range (start sample, end sample)
        };

        // DDC
        float decimationFactor = 4;
        // Note:
        // - FIR filter order must be equal 16 * decimationFactor.
        // - the filter should be symmetrical, only the upper half of the filter should be set in the hardware.
        // The below coefficients were calculate for:
        // - cutoff frequency 6 MHz,
        // - decimation factor 4, filter order 64,
        // - filter type: Hamming window.
        // E.g. in Python:
        // import scipy.signal
        // fc = 6e6
        // dec_f = 4
        // filter_order = dec_f*16
        // coefficients = scipy.signal.firwin(filter_order, fc, fs=65e6)[filter_order//2:]
        // Where 65 MHz is the us4R-lite sampling frequency.
        std::vector<float> ddcFIRCoefficients = {
            0.18234651659672152, 0.16172486625099816, 0.12487982587460944, 0.07944398046616387,
            0.03430398844523893, -0.0026133185074908405, -0.026157255063119715, -0.034889180817011325,
            -0.030945327136370222, -0.018965928416555058, -0.004494915529298055, 0.007628287152588109,
            0.014419713683593693, 0.015175743942293598, 0.011161684805841312, 0.00478318006743135,
            -0.001412573813589476, -0.005562384563359233, -0.006912138093338076, -0.005787361358840273,
            -0.0032172403273668768, -0.0004159330245921233, 0.0016683945062905931, 0.0025961471738894463,
            0.0024366998597999934, 0.0015898600953486795, 0.0005435013516173024, -0.0003223898280114102,
            -0.0008232837583015619, -0.0009500466921633298, -0.000789093632050986, -0.00044401971096737745
        };

        ::arrus::ops::us4r::DigitalDownConversion ddcOp(txFrequency, ddcFIRCoefficients, 4);
        auto result = upload(session.get(), seq, arrayModels, ddcOp);
        // Get upload results:
        // - RF buffer, which will be filled by Us4OEMS after the session is started.
        auto buffer = std::static_pointer_cast<DataBuffer>(std::get<0>(result));
        NdArrayDef outputDef = std::get<1>(result);
        auto metadata = std::get<2>(result);

        PipelineRunner runner {
            outputDef,
            metadata,
            // Processing steps to be performed on GPU.
            Pipeline{{
                RemapToLogicalOrder2{},
                ToComplex{},
                Real{}
//                EnvelopeDetection{}
                // Angle{},
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

        us4r->setVoltage(5);
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

