#include "menu.h"

#include <iostream>
#include "arrus/core/api/arrus.h"
#include "constants.h"
#include "imaging/pwi.h"
#include "common.h"

using namespace ::arrus::devices;

void setVoltage(Us4R *us4r) {
    try {
        unsigned voltage = 5;
        std::cout << "Please provide the voltage to set [V]" << std::endl;
        std::cin >> voltage;
        us4r->setVoltage(voltage);
    } catch(const arrus::IllegalArgumentException& e) {
        std::cerr << e.what() << std::endl;
    }
}

void setActiveTermination(Us4R *us4r) {
    try {
        unsigned short activeTermination = 50;
        std::cout << "Please provide active termination value [Ohm]" << std::endl;
        std::cin >> activeTermination;
        us4r->setActiveTermination(activeTermination);
    } catch(const arrus::IllegalArgumentException& e) {
        std::cerr << e.what() << std::endl;
    }
}

void setLpfCutoff(Us4R *us4r) {
    try {
        unsigned value = 15000000;
        std::cout << "Please provide LPF cutoff [Hz]" << std::endl;
        std::cin >> value;
        us4r->setLpfCutoff(value);
    } catch(const arrus::IllegalArgumentException& e) {
        std::cerr << e.what() << std::endl;
    }
}

void setDtgc(Us4R *us4r) {
    try {
        unsigned short value = 42;
        std::cout << "Please provide DTGC attenuation [dB]" << std::endl;
        std::cin >> value;
        us4r->setTgcCurve({});
        us4r->setDtgcAttenuation(value);
    } catch(const arrus::IllegalArgumentException& e) {
        std::cerr << e.what() << std::endl;
    }
}

void setPgaGain(Us4R *us4r) {
    try {
        unsigned short value = 30;
        std::cout << "Please provide PGA gain [dB]" << std::endl;
        std::cin >> value;
        us4r->setPgaGain(value);
    } catch(const arrus::IllegalArgumentException& e) {
        std::cerr << e.what() << std::endl;
    }
}

void setLnaGain(Us4R *us4r) {
    try {
        unsigned short value = 24;
        std::cout << "Please provide LNA gain [dB]" << std::endl;
        std::cin >> value;
        us4r->setLnaGain(value);
    } catch(const arrus::IllegalArgumentException& e) {
        std::cerr << e.what() << std::endl;
    }
}

void setLinearTgc(Us4R *us4r, const ::arrus::imaging::PwiSequence &seq) {
    try {
        float tgcStart, tgcSlope;
        std::cout << "TGC curve start value [dB]" << std::endl;
        std::cin >> tgcStart;
        std::cout << "TGC curve slope [dB/m]" << std::endl;
        std::cin >> tgcSlope;
        std::vector<float> tgcCurve = getLinearTGCCurve(
            tgcStart, tgcSlope, SAMPLING_FREQUENCY, seq.getSpeedOfSound(), seq.getSampleRange().second);

        std::cout << "Applying TGC curve: " << std::endl;
        for(auto &value: tgcCurve) {
            std::cout << value << ", ";
        }
        std::cout << std::endl;
        us4r->setDtgcAttenuation(std::nullopt);
        us4r->setTgcCurve(tgcCurve, false);
    }
    catch(const arrus::IllegalArgumentException &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
    }
}

void runMainMenu(Us4R *us4r, const ::arrus::imaging::PwiSequence &sequence) {
    // Here the main thread waits until user presses 'q' button.
    // All the processing and displaying is done by callback threads.
    char lastChar = 0;
    while (lastChar != 'q') {
        std::cout << "Menu: " << std::endl;
        std::cout << "v - set voltage" << std::endl;
        std::cout << "t - set linear tgc (turns off digital TGC)" << std::endl;
        std::cout << "a - sets active termination" << std::endl;
        std::cout << "f - sets LPF cutoff" << std::endl;
        std::cout << "p - sets PGA gain" << std::endl;
        std::cout << "l - sets LNA gain" << std::endl;
        std::cout << "d - sets DTGC attenuation (turns off analog TGC)" << std::endl;
        std::cout << "q - quit" << std::endl;
        std::cout << "Choose an option and press enter" << std::endl;
        std::cin >> lastChar;
        switch(lastChar) {
            case 'v':
                // Set voltage
                setVoltage(us4r);
                break;
            case 't':
                // Set TGC curve (linear)
                setLinearTgc(us4r, sequence);
                break;
            case 'a':
                setActiveTermination(us4r);
                break;
            case 'f':
                setLpfCutoff(us4r);
                break;
            case 'd':
                setDtgc(us4r);
                break;
            case 'l':
                setLnaGain(us4r);
                break;
            case 'p':
                setPgaGain(us4r);
                break;
            case 'q':
                std::cout << "Stopping application" << std::endl;
                break;
            default:
                std::cerr << "Unknown command: " << lastChar << std::endl;
        }
    }
}
