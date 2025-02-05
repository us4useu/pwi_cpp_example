#include "common.h"

#include <vector>
#include <fstream>

std::vector<float> arange(float l, float r, float step) {
    std::vector<float> result;
    float currentPos = l;
    while(currentPos <= r) {
        result.push_back(currentPos);
        currentPos += step;
    }
    return result;
}

std::vector<float> linspace(float l, float r, unsigned n) {
    std::vector<float> result(n, 0.0f);
    float step = (r-l)/(float)(n-1);
    float currentAngle = l;
    for(int i = 0; i < n; ++i) {
        result[i] = currentAngle;
        currentAngle += step;
    }
    return result;
}

void writeDataToFile(const std::string& path, char* dataPtr, size_t nBytes) {
    std::ofstream file;
    file.open(path, std::ios_base::binary);
    file.write((char*)dataPtr, nBytes);
    file.close();
}

std::vector<float> getLinearTGCCurve(float tgcStart, float tgcSlope, float samplingFrequency, float speedOfSound,
                                     float sampleRangeEnd) {
    std::vector<float> tgcCurve;

    float startDepth = 300.0f/samplingFrequency*speedOfSound;
    float endDepth = sampleRangeEnd/samplingFrequency*speedOfSound;
    float tgcSamplingStep = 150.0f/samplingFrequency*speedOfSound;
    float currentDepth = startDepth;

    while(currentDepth < endDepth) {
        float tgcValue = tgcStart+tgcSlope*currentDepth;
        tgcCurve.push_back(tgcValue);
        currentDepth += tgcSamplingStep;
    }
    return tgcCurve;
}