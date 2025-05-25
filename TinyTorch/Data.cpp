/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */
#include "Data.h"

#include <cassert>
#include <fstream>
#include <memory>

#include "Logger.h"



namespace TinyTorch::data::transforms {

Tensor Normalize::process(Tensor& input) const {
  Tensor ret = Tensor::shape(input.shape());
  auto& src = input.data();
  auto& dst = ret.data();
  dst = (src - mean_) / std_;
  return ret;
}

Tensor Resize::process(cv::Mat& image) const {
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(shape_[0],
       shape_[1]), 0, 0, cv::INTER_LINEAR);
  return Tensor(std::move(TensorImpl(resized_image)));
}

}  // namespace TinyTorch::data::transforms

namespace TinyTorch::data {

constexpr auto MNIST_TRAIN_IMAGES = "train-images-idx3-ubyte";
constexpr auto MNIST_TRAIN_LABELS = "train-labels-idx1-ubyte";
constexpr auto MNIST_TEST_IMAGES = "t10k-images-idx3-ubyte";
constexpr auto MNIST_TEST_LABELS = "t10k-labels-idx1-ubyte";

DatasetMNIST::DatasetMNIST(
    const std::string& dir, MnistDataType type,
    const std::shared_ptr<transforms::Transform>& transform)
    : transform_(transform) {
  auto imagePath =
      dir + (type == TRAIN ? MNIST_TRAIN_IMAGES : MNIST_TEST_IMAGES);
  auto labelPath =
      dir + (type == TRAIN ? MNIST_TRAIN_LABELS : MNIST_TEST_LABELS);
  loadImages(imagePath);
  loadLabels(labelPath);
  size_ = std::min(images_.size(), labels_.size());
}

std::vector<Tensor> DatasetMNIST::getItem(size_t idx) {
  auto img = Tensor(images_[idx]);
  img.data().reshape_({1, height_, width_});

  auto label = Tensor::scalar(labels_[idx]);
  if (transform_) {
    img = transform_->process(img);
  }
  return {img, label};
}

int32_t DatasetMNIST::toInt32(const char* p) {
  return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) | ((p[2] & 0xff) << 8) |
         ((p[3] & 0xff) << 0);
}

void DatasetMNIST::loadImages(const std::string& path) {
  std::ifstream ifs(path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    LOGE("failed to load images from %s", path.c_str());
    return;
  }
  char p[4];

  ifs.read(p, 4);
  auto magicNumber = toInt32(p);
  assert(magicNumber == 0x803);

  ifs.read(p, 4);
  auto size = toInt32(p);
  images_.resize(size);

  ifs.read(p, 4);
  height_ = toInt32(p);

  ifs.read(p, 4);
  width_ = toInt32(p);

  char* tmp = new char[height_ * width_];
  for (int32_t i = 0; i < size; ++i) {
    images_[i].resize(height_ * width_);
    ifs.read(tmp, height_ * width_);
    float* dataPtr = &images_[i][0];
    for (int32_t j = 0; j < height_ * width_; ++j) {
      dataPtr[j] = (float)((uint8_t)tmp[j]) / 255.0f;
    }
  }
  delete[] tmp;

  ifs.close();
}

void DatasetMNIST::loadLabels(const std::string& path) {
  std::ifstream ifs(path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    LOGE("failed to load labels from %s", path.c_str());
    return;
  }
  char p[4];
  ifs.read(p, 4);
  auto magicNumber = toInt32(p);
  assert(magicNumber == 0x801);

  ifs.read(p, 4);
  auto size = toInt32(p);
  labels_.resize(size);
  for (int32_t i = 0; i < size; ++i) {
    ifs.read(p, 1);
    labels_[i] = (float)(p[0]);
  }

  ifs.close();
}

cv::Mat readImage(const std::string& image_path) {
    #ifdef USE_OPENCV
      cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
      if (image.empty()) {
        throw std::runtime_error("Failed to open image file: "
                                 + image_path);
      }
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
      return image;
    #else
      throw std::out_of_range("We only use opencv to read image, "
      "please install opencv");
    #endif
}

std::vector<Tensor> DatasetYOLO::getItem(size_t idx) {
  if (idx >= size_) {
    throw std::out_of_range("Index out of range");
  }
  const std::string& line = annotation_lines_[idx];
  std::istringstream iss(line);
  std::string image_path;
  iss >> image_path;
  cv::Mat original_image = readImage(image_path);
  int original_width = original_image.cols;
  int original_height = original_image.rows;

  Tensor image_t;
  if (transform_) {
    image_t = transform_->process(original_image);
  } else {
    image_t = Tensor(std::move(TensorImpl(original_image)));
  }

  width_ = image_t.shape()[2];
  height_ = image_t.shape()[1];

  float width_ratio = static_cast<float>(width_) / original_width;
  float height_ratio = static_cast<float>(height_) / original_height;

  std::vector<std::vector<float>> labels;
  std::string token;
  while (iss >> token) {
    std::istringstream token_iss(token);
    std::string value;
    std::vector<float> box;
    while (std::getline(token_iss, value, ',')) {
      box.push_back(std::stof(value));
    }
    if (box.size() == 5) { // class_id, x_center, y_center, width, height
      box[1] *= width_ratio;  // x_center
      box[2] *= height_ratio; // y_center
      box[3] *= width_ratio;  // width
      box[4] *= height_ratio; // height
      labels.push_back(box);
    }
  }

  const int default_invalid_value = -1;
  while (labels.size() < max_targets_) {
    labels.push_back({static_cast<float>(default_invalid_value),
                      static_cast<float>(default_invalid_value),
                      static_cast<float>(default_invalid_value),
                      static_cast<float>(default_invalid_value),
                      static_cast<float>(default_invalid_value)});
  }
  return {image_t, Tensor(labels)};
}
DatasetYOLO::DatasetYOLO(
    const std::string& annotation_path, YOLODataType type,
    const std::shared_ptr<transforms::Transform>& transform)
    : transform_(transform){

  std::ifstream infile(annotation_path);
  if (!infile.is_open()) {
    throw std::runtime_error("Can't open the file" + annotation_path);
  }
  std::string line;
  if (std::getline(infile, line)) {
    std::istringstream iss(line);
    iss >> max_targets_ >> num_classes_;
    if (iss.fail() || max_targets_ <= 0 || num_classes_ <= 0) {
      throw std::runtime_error("invaild max_targets or num_class");
    }
  } else {
    throw std::runtime_error("annotation file is not correct");
  }
  while (std::getline(infile, line)) {
    annotation_lines_.push_back(line);
  }
  infile.close();
  size_ = annotation_lines_.size();
}

}  // namespace TinyTorch::data


