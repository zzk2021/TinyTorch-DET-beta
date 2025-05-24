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


std::vector<std::string> DatasetYOLO::readAnnotationFile(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open annotation file: " + path);
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(ifs, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}


Tensor readImage(const std::string& image_path) {
    std::ifstream ifs(image_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open image file: " + image_path);
    }
    std::vector<std::vector<float>> image_data;
    std::string line;
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        std::vector<float> row;
        float pixel_value;
        while (iss >> pixel_value) {
            row.push_back(pixel_value);
        }
        if (!row.empty()) {
            image_data.push_back(row);
        }
    }

    if (image_data.empty() || image_data[0].empty()) {
        throw std::runtime_error("Invalid image data in file: " + image_path);
    }

    int height = image_data.size();
    int width_times_channels = image_data[0].size();

    int channels = 3;
    int width = width_times_channels / channels;

    std::vector<std::vector<std::vector<float>>> nested_data(channels,
        std::vector<std::vector<float>>(height, std::vector<float>(width)));

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                nested_data[c][h][w] = image_data[h][w * channels + c];
            }
        }
    }
    return Tensor(nested_data);
}

 std::vector<Tensor> DatasetYOLO::getItem(size_t idx) {
  if (idx >= size_) {
    throw std::out_of_range("Index out of range");
  }
  const std::string& line = annotation_lines_[idx];
  std::istringstream iss(line);
  std::string image_path;
  iss >> image_path;
  Tensor image = readImage(image_path);
  if (transform_) {
    image = transform_->process(image);
  }
  std::vector<std::vector<float>> labels;
  std::string token;
  while (iss >> token) {
    std::istringstream token_iss(token);
    std::string value;
    std::vector<float> box;
    while (std::getline(token_iss, value, ',')) {
      box.push_back(std::stof(value));
    }
    if (box.size() == 5) { // x_center, y_center, width, height, class_id
      labels.push_back(box);
    }
  }
  return {image, Tensor(labels)}; // 假设 Tensor 可以接受 vector<vector<float>>
}

DatasetYOLO::DatasetYOLO(
    const std::string& annotation_lines, YOLODataType type,
    const std::shared_ptr<transforms::Transform>& transform)
    : transform_(transform) {

  size_ = std::min(images_.size(), labels_.size());
}

}  // namespace TinyTorch::data


