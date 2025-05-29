#include "tools.h"

json loadConfig(std::string path) {
  std::ifstream file(path);
  if (!file.is_open()) throw std::runtime_error("Failed to open config file");
  json j;
  file >> j;
  return j;
}

fs::path currentPath(){
    return fs::current_path();
}