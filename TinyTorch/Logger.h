/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <mutex>

namespace TinyTorch {

// clang-format off
#define LOGI(...) TinyTorch::Logger::log(TinyTorch::LOG_INFO,     __FILE__, __LINE__, __VA_ARGS__)
#define LOGD(...) TinyTorch::Logger::log(TinyTorch::LOG_DEBUG,    __FILE__, __LINE__, __VA_ARGS__)
#define LOGW(...) TinyTorch::Logger::log(TinyTorch::LOG_WARNING,  __FILE__, __LINE__, __VA_ARGS__)
#define LOGE(...) TinyTorch::Logger::log(TinyTorch::LOG_ERROR,    __FILE__, __LINE__, __VA_ARGS__)
// clang-format on

static constexpr int MAX_LOG_LENGTH = 1024;
typedef void (*LogFunc)(void *context, int level, const char *msg);

enum LogLevel {
  LOG_INFO,
  LOG_DEBUG,
  LOG_WARNING,
  LOG_ERROR,
};

class Logger {
 public:
  static void log(LogLevel level, const char *file, int line,
                  const char *message, ...);

 private:
  static void *logContext_;
  static LogFunc logFunc_;
  static LogLevel minLevel_;

  static char buf_[MAX_LOG_LENGTH];
  static std::mutex mutex_;
};

}  // namespace TinyTorch
