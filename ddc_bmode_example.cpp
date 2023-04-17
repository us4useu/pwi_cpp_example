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
#include "imaging/ops/Imag.h"
#include "imaging/ops/ToBMode.h"
#include "imaging/ops/Phase.h"
#include "imaging/ops/ReconstructHri.h"

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

        // Decimation factor 4
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
        // Decimation factor 8
//        std::vector<float> ddcFIRCoefficients = {
//              0.18221508637342165, 0.16216728768173705, 0.12609126535675264, 0.08105520413870602,
//              0.035492307126482145, -0.0027518412706919715, -0.028136722264028014, -0.03848370355254714,
//              -0.035139444223612364, -0.022261821923196916, -0.0054769951821222376, 0.009692034248630653,
//              0.019192949548113916, 0.02126533024300503, 0.0165520470654214, 0.007548095776489814,
//              -0.0023860429004200754, -0.010120030943892883, -0.0136351669875016, -0.012465051394079723,
//              -0.007621653465682795, -0.0010920316133656946, 0.0048918473651834, 0.008563429051726431,
//              0.009099948580800946, 0.006751889261376747, 0.00262677115977431, -0.0017637399099874565,
//              -0.005024291728682894, -0.006294003318012586, -0.005444769445704029, -0.0030252224866296174,
//              -3.767547385180949e-18, 0.0026076029156972633, 0.004044414937731745, 0.0040272705046610615,
//              0.002767507691238393, 0.0008356170029311679, -0.0010692693890346477, -0.002358409470046541,
//              -0.0027233311282272156, -0.0021918733040187884, -0.001068768623728969, 0.00020319936149566808,
//              0.001204871718065625, 0.0016695980904884195, 0.001542833868699534, 0.0009642475408967266,
//              0.00019078008894664097, -0.000504584792768131, -0.0009214647858365811, -0.0009817740503059637,
//              -0.0007315849837409322, -0.00030357673152717326, 0.0001402546598517935, 0.0004634752876123044,
//              0.0005909791256731553, 0.0005187960922712293, 0.0003009774586373571, 2.3023178831062144e-05,
//              -0.0002271235130510075, -0.00038203900064714004, -0.00040776375778725325, -0.0003068463604729795
//        };
        // Simple averaging filter.
//        int filterOrder = (int)decimationFactor*16;
//        std::vector<float> ddcFIRCoefficients(filterOrder, 1.0f);

        ::arrus::ops::us4r::DigitalDownConversion ddcOp(txFrequency, ddcFIRCoefficients, decimationFactor);
        auto result = upload(session.get(), seq, arrayModels, ddcOp);
        // Get upload results:
        // - RF buffer, which will be filled by Us4OEMS after the session is started.
        auto buffer = std::static_pointer_cast<DataBuffer>(std::get<0>(result));
        NdArrayDef outputDef = std::get<1>(result);
        auto metadata = std::get<2>(result);

        constexpr float X_LEFT_BORDER = -19.0e-3f, X_RIGHT_BORDER = 19.0e-3f, X_STEP = 0.1e-3;
        constexpr float Z_LEFT_BORDER = 5e-3f, Z_RIGHT_BORDER = 42.5e-3f, Z_STEP = 0.1e-3;
        auto xGrid = ::arrus_example_imaging::NdArray::asarray(arange(X_LEFT_BORDER, X_RIGHT_BORDER, X_STEP));
        auto zGrid = ::arrus_example_imaging::NdArray::asarray(arange(Z_LEFT_BORDER, Z_RIGHT_BORDER, Z_STEP));

        PipelineRunner runner {
            outputDef,
            metadata,
            // Processing steps to be performed on GPU.
            Pipeline{{
                RemapToLogicalOrder2{},
                ToComplex{},
                ReconstructHri(xGrid, zGrid),
                EnvelopeDetection(),
                ToBMode(::arrus_example_imaging::NdArray::asarray<float>(0),
                        ::arrus_example_imaging::NdArray::asarray<float>(80))
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

