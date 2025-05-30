#pragma once

#ifdef DEBUG
   #include <Windows.h>
   #include <DbgHelp.h>
   #include <iostream>
   #include <sstream>
   #include <mutex>



   void InitializeDbgHelp();

   std::string GetStackTrace(int skip_frames = 0, int max_frames = 62);

    inline void HandleAssertionFailure(
        const char* expr,
        const char* file,
        int line,
        const char* func)
    {
        std::string stack = GetStackTrace(1, 8);
        std::ostringstream msg;
        msg << "Assertion failed: " << expr << "\n"
            << "File: " << file << "\n"
            << "Line: " << line << "\n"
            << "Function: " << func << "\n"
            << stack;
        std::cerr << msg.str() << std::endl;
        if (IsDebuggerPresent()) {
            OutputDebugStringA(msg.str().c_str());
            DebugBreak();
        }
        abort();
    }

   #define ASSERT(expr) \
    do { \
        if (!(expr)) { \
            HandleAssertionFailure(#expr, __FILE__, __LINE__, __func__); \
        } \
    } while(0)
#else
   #define ASSERT(expr) \
        do { \
            if (!(expr)) { \
                std::cerr << "Assertion failed: " << #expr << "\n" \
                          << "File: " << __FILE__ << "\n" \
                          << "Line: " << __LINE__ << "\n" \
                          << "Function: " << __func__ << "\n"; \
                assert(false && #expr); \
            } \
        } while(0)
#endif