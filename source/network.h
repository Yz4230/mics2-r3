#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <torch/script.h>

#include <unordered_map>
#include <vector>

#include "position.h"

namespace Network {
extern torch::jit::script::Module *model;

class LinearNetwork {
 private:
  std::vector<double> weights;
  LinearNetwork(const std::vector<double> &weights);
  double sigmoid(double x);

 public:
  static LinearNetwork from_weights_file(const std::string &filename);
  double predict(const std::vector<int32_t> &inputs);
};

extern LinearNetwork *network;
extern std::unordered_map<int32_t, int32_t> key_to_index;

std::unordered_map<int32_t, int32_t> from_key_to_index_file(
    const std::string &filename);

}  // namespace Network

#endif  // _NETWORK_H_
