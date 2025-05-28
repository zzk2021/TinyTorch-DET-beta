
#include "../Function.h"
#include "Function_explore.h"

namespace TinyTorch {

std::vector<int> FindBestAnchors(const std::vector<std::vector<float>>& iou_matrix) {
    std::vector<int> best_anchors = {};
    for (const auto& row : iou_matrix) {
        auto max_iter = std::max_element(row.begin(), row.end());
        int index = static_cast<int>(std::distance(row.begin(), max_iter));
        best_anchors.push_back(index);
    }
    return best_anchors;
}

std::vector<std::vector<float>> ConvertToCorners(const std::vector<std::vector<float>>& boxes) {
    size_t num_boxes = boxes.size();
    std::vector<std::vector<float>> corners(num_boxes, std::vector<float>(4));
    for (size_t i = 0; i < num_boxes; ++i) {
        float cx = boxes[i][0];
        float cy = boxes[i][1];
        float w  = boxes[i][2];
        float h  = boxes[i][3];
        corners[i][0] = cx - w / 2.0f; // x1
        corners[i][1] = cy - h / 2.0f; // y1
        corners[i][2] = cx + w / 2.0f; // x2
        corners[i][3] = cy + h / 2.0f; // y2
    }
    return corners;
}

std::vector<std::vector<float>> CalculateIOU(const std::vector<std::vector<float>>& box_a, const std::vector<std::vector<float>>& box_b) {
  if (box_a.size() == 0 || box_b.size() == 0){
      std::vector<std::vector<float>> p = {};
      return p;
  }
  std::vector<std::vector<float>> a_corners = ConvertToCorners(box_a);
  std::vector<std::vector<float>> b_corners = ConvertToCorners(box_b);

  size_t A = a_corners.size();
  size_t B = b_corners.size();

  std::vector<std::vector<float>> iou(A, std::vector<float>(B, 0.0f));

  for (size_t i = 0; i < A; ++i) {
    float a_x1 = a_corners[i][0];
    float a_y1 = a_corners[i][1];
    float a_x2 = a_corners[i][2];
    float a_y2 = a_corners[i][3];
    float a_area = (a_x2 - a_x1) * (a_y2 - a_y1);

    for (size_t j = 0; j < B; ++j) {
      float b_x1 = b_corners[j][0];
      float b_y1 = b_corners[j][1];
      float b_x2 = b_corners[j][2];
      float b_y2 = b_corners[j][3];
      float b_area = (b_x2 - b_x1) * (b_y2 - b_y1);

      float inter_x1 = std::max(a_x1, b_x1);
      float inter_y1 = std::max(a_y1, b_y1);
      float inter_x2 = std::min(a_x2, b_x2);
      float inter_y2 = std::min(a_y2, b_y2);

      float inter_width  = std::max(0.0f, inter_x2 - inter_x1);
      float inter_height = std::max(0.0f, inter_y2 - inter_y1);
      float inter_area   = inter_width * inter_height;

      float union_area = a_area + b_area - inter_area;

      if (union_area > 0.0f) {
        iou[i][j] = inter_area / union_area;
      } else {
        iou[i][j] = 0.0f;
      }
      }
    }
    return iou;
}

Tensor Function::giou(const Tensor& a, const Tensor& b) {
    //        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    //        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
  return std::make_shared<FuncGIOU>()->callForward({&a, &b});
}

TensorImpl FuncGIOU::forward(const std::vector<const Tensor *> &inputs)  {
    throw std::runtime_error("We have not implement in CPU yet");
}
std::vector<TensorImpl> FuncGIOU::backward(const TinyTorch::TensorImpl &grad)  {
    throw std::runtime_error("We have not implement in CPU yet");
}

}
