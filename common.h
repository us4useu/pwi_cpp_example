#ifndef CPP_EXAMPLE_COMMON_H
#define CPP_EXAMPLE_COMMON_H

#include <string>
#include <vector>

std::vector<float> arange(float l, float r, float step);
std::vector<float> linspace(float l, float r, unsigned n);
void writeDataToFile(const std::string& path, char* dataPtr, size_t nBytes);
std::vector<float> getLinearTGCCurve(float tgcStart, float tgcSlope, float samplingFrequency, float speedOfSound,
                                     float sampleRangeEnd);


#endif //CPP_EXAMPLE_COMMON_H
