#ifndef CPP_EXAMPLE_MYCUSTOMLOGGER_H
#define CPP_EXAMPLE_MYCUSTOMLOGGER_H

#include <iostream>
#include "arrus/core/api/common.h"

/**
 * A simple custom logger which just prints a given message to stderr.
 */
class MyCustomLogger : public ::arrus::Logger {
public:
    explicit MyCustomLogger(arrus::LogSeverity loggingLevel, std::string deviceId)
    : loggingLevel(loggingLevel), deviceId(std::move(deviceId)) {}

    /**
     * Prints a message with given severity to stderr.
     * If the selected severity is, nothing is printed on console.
     *
     * @param severity message severity
     * @param msg message to print
     */
    void
    log(const arrus::LogSeverity severity, const std::string &msg) override {
        if(severity >= loggingLevel) {
            std::cerr << "[" << deviceId << "]" << "[" << severityToString(severity) << "]: " << msg << std::endl;
        }
    }

    void
    setAttribute(const std::string &key, const std::string &value) override {
        if(key == "DeviceId") {
            deviceId = value;
        }
    }
private:

    std::string severityToString(const arrus::LogSeverity severity) {
        switch(severity) {
            case arrus::LogSeverity::TRACE: return "trace";
            case arrus::LogSeverity::DEBUG: return "debug";
            case arrus::LogSeverity::INFO: return "info";
            case arrus::LogSeverity::WARNING: return "warning";
            case arrus::LogSeverity::ERROR: return "error";
            case arrus::LogSeverity::FATAL: return "fatal";
            default: return "unknown";
        }
    }
    arrus::LogSeverity loggingLevel;
    std::string deviceId{"unknown device"};
};

#endif //CPP_EXAMPLE_MYCUSTOMLOGGER_H
