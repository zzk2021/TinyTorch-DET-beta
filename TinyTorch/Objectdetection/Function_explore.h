#pragma once

#include "../Function.h"


namespace TinyTorch {

class FuncGIOU : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_GIOU)
};

std::vector<std::vector<float>> CalculateIOU(const std::vector<std::vector<float>>& box_a, const std::vector<std::vector<float>>& box_b);
std::vector<std::vector<float>> ConvertToCorners(const std::vector<std::vector<float>>& boxes);
std::vector<int> FindBestAnchors(const std::vector<std::vector<float>>& iou_matrix);
}