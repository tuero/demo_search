// File: logging.h
// Description: Logging setup

#ifndef HPTS_COMMON_LOGGING_H_
#define HPTS_COMMON_LOGGING_H_

#include <string>

namespace hpts {

/**
 * Initialize the file and terminal loggers
 * @param path The directory which the experiment output resides
 * @param console_only Flag to only log to console
 * @param postfix Postfix for logger name file
 */
void init_loggers(const std::string &path, bool console_only = false, const std::string &postfix = "");

/**
 * Log the invoked command used to run the current program
 * @param argc Number of arguments
 * @param argv char array of params
 */
void log_flags(int argc, char **argv);

/**
 * Flush the logs
 */
void log_flush();

/**
 * Close all loggers
 */
void close_loggers();

}    // namespace hpts

#endif    // HPTS_COMMON_LOGGING_H_
