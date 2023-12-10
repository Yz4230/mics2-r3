
#include "network.h"

#include <torch/script.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "position.h"

namespace Network {
// 先手が勝つ確率
torch::jit::script::Module *model = nullptr;

LinearNetwork::LinearNetwork(const std::vector<double> &weights)
    : weights(weights) {}

/**
 * @brief Load weights from file
 * @deprecated Use Network::model instead
 *
 * @param filename
 * @return LinearNetwork
 */
LinearNetwork LinearNetwork::from_weights_file(const std::string &filename) {
  std::ifstream file(filename);
  // check existence
  if (!file.good()) {
    std::cerr << "File not found: " << filename << std::endl;
    exit(1);
  }

  std::vector<double> weights(46800);
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    // format: {index}:{weight}
    size_t sep_index = line.find(":");
    int index = std::stoi(line.substr(0, sep_index));
    double weight = std::stod(line.substr(sep_index + 1));
    weights[index] = weight;
  }
  return LinearNetwork(weights);
}

double LinearNetwork::sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double LinearNetwork::predict(const std::vector<int32_t> &keys) {
  double sum = 0.0;
  for (auto key : keys) {
    sum += weights[key_to_index[key]];
  }
  return sigmoid(sum);
}

LinearNetwork *network;
std::unordered_map<int32_t, int32_t> key_to_index =
    std::unordered_map<int32_t, int32_t>{};

std::unordered_map<int32_t, int32_t> from_key_to_index_file(
    const std::string &filename) {
  std::ifstream file(filename);
  // check existence
  if (!file.good()) {
    std::cerr << "File not found: " << filename << std::endl;
    exit(1);
  }

  std::unordered_map<int32_t, int32_t> key_to_index;
  std::string line;

  while (std::getline(file, line)) {
    if (line.empty()) continue;
    // format: {key}:{index}
    size_t sep_index = line.find(":");
    int32_t key = std::stoi(line.substr(0, sep_index));
    int32_t index = std::stoi(line.substr(sep_index + 1));
    key_to_index[key] = index;
  }

  return key_to_index;
}

}  // namespace Network
