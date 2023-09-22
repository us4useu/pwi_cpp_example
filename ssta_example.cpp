#include <iostream>
#include <thread>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
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

#include "imaging/ops/RemapToLogicalOrder.h"
#include "imaging/ops/Transpose.h"
#include "imaging/ops/BandpassFilter.h"
#include "imaging/ops/DigitalDownConversion.h"
#include "imaging/ops/ReconstructHri.h"
#include "imaging/ops/EnvelopeDetection.h"
#include "imaging/ops/ToBMode.h"

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

constexpr size_t apertureSize = 128;
constexpr size_t nTx = 128;
constexpr size_t nSamples = 2048;
bool saveOnDisk = true;

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

        // The below will give the total number of channels in the system (256 for us4R-lite).
        auto nChannels = us4r->getProbe(0)->getModel().getNumberOfElements().product();

        ::arrus::BitMask rxApertureOY(nChannels, false);
        // RX aperture [128, 256) -- OY elements.
        for(int i = 128; i < 256; ++i) {rxApertureOY[i] = true;}

        Pulse pulse(6e6, 2, false);
        ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, nSamples};
        float speedOfSound = 1500; // [m/s]

        std::vector<TxRx> txrxs;
        // TX: elements from 0 to 128 (OX), RX: rxApertureOY
        for(int tx = 0; tx < 128; ++tx) {
            std::vector<float> delays(nChannels, 0.0f);
            arrus::BitMask txAperture(nChannels, false);
            txAperture[tx] = true;
            txrxs.emplace_back(Tx(txAperture, delays, pulse), Rx(rxApertureOY, sampleRange), 200e-6f);
        }

        TxRxSequence seq(txrxs, {}, TxRxSequence::NO_SRI);
        DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 2};
        Scheme scheme(seq, 2, outputBuffer, Scheme::WorkMode::HOST);
        auto result = session->upload(scheme);
        ::arrus_example_imaging::MetadataBuilder metadataBuilder;
        // Required by RemapToLogicalOrder.
        metadataBuilder.addObject(
            "frameChannelMapping",
            result.getConstMetadata()->get<::arrus::devices::FrameChannelMapping>("frameChannelMapping"));
        // Required by RemapToLogicalOrder.
        metadataBuilder.addObject("rawSequence", std::make_shared<TxRxSequence>(seq));

        auto buffer = std::static_pointer_cast<DataBuffer>(result.getBuffer());
        auto outputShape = buffer->getElement(0)->getData().getShape().getValues();
        NdArrayDef outputDef(outputShape, ::arrus_example_imaging::DataType::INT16);
        auto metadata = metadataBuilder.buildSharedPtr();

        PipelineRunner runner {
            outputDef,
            metadata,
            // Processing steps to be performed on GPU.
            Pipeline({
                RemapToLogicalOrder{},
            })
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
                    [](void* input) {
                        if(saveOnDisk) {
                            std::ofstream fout;
                            fout.open("data.bin", std::ios::binary | std::ios::out);
                            fout.write((char*)input, nTx*nSamples*apertureSize*sizeof(int16_t));
                            fout.close();
                            saveOnDisk = false;
                            std::cout << "Data saved to 'data.bin'" << std::endl;
                        }
                        mainDisplay.update(input);
                    }
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
        runMainMenu(us4r, speedOfSound, sampleRange.second);
        // Exit after stopping the scheme.
        session->stopScheme();
        std::cout << "Done" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

