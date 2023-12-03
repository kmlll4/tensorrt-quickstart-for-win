#ifndef TENSORRT_SAMPLE_COMMON_LOGGER_H
#define TENSORRT_SAMPLE_COMMON_LOGGER_H

#include "logging.h"

namespace sample
{
extern Logger gLogger;
extern LogStreamConsumer gLogVerbose;
extern LogStreamConsumer gLogInfo;
extern LogStreamConsumer gLogWarning;
extern LogStreamConsumer gLogError;
extern LogStreamConsumer gLogFatal;

void setReportableSeverity(Logger::Severity severity);
} // namespace sample

#endif // TENSORRT_SAMPLE_COMMON_LOGGER_H
