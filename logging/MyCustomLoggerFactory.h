#ifndef CPP_EXAMPLE_MYCUSTOMLOGGERFACTORY_H
#define CPP_EXAMPLE_MYCUSTOMLOGGERFACTORY_H

#include "MyCustomLogger.h"
#include "arrus/core/api/common/logging.h"

/**
 * A custom LoggerFactory.
 * The instance of this class is by ARRUS to create arrus::Logger instances.
 */
class MyCustomLoggerFactory: public ::arrus::LoggerFactory {
public:
    explicit MyCustomLoggerFactory(arrus::LogSeverity severityLevel)
    : severityLevel(severityLevel) {}

    /**
     * Returns a new Logger instance.
     */
    arrus::Logger::Handle getLogger() override {
        return getLogger({});

    }

    arrus::Logger::Handle getLogger(const std::vector<arrus::Logger::Attribute> &attributes) override {
        std::string deviceId{"-"};
        for(auto &[key, value]: attributes) {
            std::cout << "key: " << key << std::endl;
            if(key == "DeviceId") {
                deviceId = value;
            }
        }
        return std::make_unique<MyCustomLogger>(severityLevel, deviceId);
    }
private:
    ::arrus::LogSeverity severityLevel;
};


#endif //CPP_EXAMPLE_MYCUSTOMLOGGERFACTORY_H
