#pragma once
#include "traceback.h"

#ifdef DEBUG
#pragma comment(lib, "DbgHelp.lib")
std::once_flag dbghelp_init_flag;
void InitializeDbgHelp() {
    std::call_once(dbghelp_init_flag, [] {
        SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
        SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    });
}

std::string GetStackTrace(int skip_frames, int max_frames) {
    InitializeDbgHelp();

    void* stack[62];
    HANDLE process = GetCurrentProcess();
    USHORT frames = CaptureStackBackTrace(skip_frames + 1, max_frames, stack, nullptr);

    SYMBOL_INFO* symbol = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

    IMAGEHLP_LINE64 line = {0};
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);

    std::ostringstream oss;
    DWORD displacement;

    oss << "Stack Trace:\n";
    for (USHORT i = 0; i < frames; i++) {
        DWORD64 address = (DWORD64)(stack[i]);
        SymFromAddr(process, address, 0, symbol);

        if (SymGetLineFromAddr64(process, address, &displacement, &line)) {
            oss << "[" << i << "] " << symbol->Name << " ("
                << line.FileName << ":" << line.LineNumber << ")\n";
        } else {
            oss << "[" << i << "] " << symbol->Name << " (0x"
                << std::hex << address << std::dec << ")\n";
        }
    }

    free(symbol);
    return oss.str();
}
#endif
