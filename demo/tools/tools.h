#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>


using json = nlohmann::json;
namespace fs = std::filesystem;

json loadConfig(std::string path);
fs::path currentPath() ;