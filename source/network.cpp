#include "network.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "position.h"

namespace Network {

LinearNetwork::LinearNetwork(const std::vector<double> &weights)
    : weights(weights) {}

LinearNetwork LinearNetwork::from_weights_file(const std::string &filename) {
  std::ifstream file(filename);
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

std::vector<int32_t> position_to_keys(const Position &pos) {
  static const std::unordered_map<Piece, int32_t> piece_to_index = {
      {Piece::B_KING, 0}, {Piece::B_ROOK, 1},    {Piece::B_BISHOP, 2},
      {Piece::B_GOLD, 3}, {Piece::B_SILVER, 4},  {Piece::B_PAWN, 5},
      {Piece::W_KING, 6}, {Piece::W_ROOK, 7},    {Piece::W_BISHOP, 8},
      {Piece::W_GOLD, 9}, {Piece::W_SILVER, 10}, {Piece::W_PAWN, 11},
  };

  static const std::unordered_map<Piece, Piece> unpromoted_piece = {
      {Piece::B_KING, Piece::B_KING},
      {Piece::B_DRAGON, Piece::B_ROOK},
      {Piece::B_ROOK, Piece::B_ROOK},
      {Piece::B_HORSE, Piece::B_BISHOP},
      {Piece::B_BISHOP, Piece::B_BISHOP},
      {Piece::B_GOLD, Piece::B_GOLD},
      {Piece::B_PRO_SILVER, Piece::B_SILVER},
      {Piece::B_SILVER, Piece::B_SILVER},
      {Piece::B_PRO_PAWN, Piece::B_PAWN},
      {Piece::B_PAWN, Piece::B_PAWN},
      {Piece::W_KING, Piece::W_KING},
      {Piece::W_DRAGON, Piece::W_ROOK},
      {Piece::W_ROOK, Piece::W_ROOK},
      {Piece::W_HORSE, Piece::W_BISHOP},
      {Piece::W_BISHOP, Piece::W_BISHOP},
      {Piece::W_GOLD, Piece::W_GOLD},
      {Piece::W_PRO_SILVER, Piece::W_SILVER},
      {Piece::W_SILVER, Piece::W_SILVER},
      {Piece::W_PRO_PAWN, Piece::W_PAWN},
      {Piece::W_PAWN, Piece::W_PAWN},
  };

  // key: `|sq1(5bit)|sq2(5bit)|pc1(4bit)|pc2(4bit)| = 18bit`
  std::vector<int32_t> keys;

  constexpr int32_t NB = 25;
  for (int32_t i = 0; i < NB; i++)
    for (int32_t j = i + 1; j < NB; j++) {
      Square sq1 = Square(i);
      Square sq2 = Square(j);
      Piece pc1 = pos.piece_on(sq1);
      Piece pc2 = pos.piece_on(sq2);

      if (pc1 == Piece::NO_PIECE || pc2 == Piece::NO_PIECE) continue;
      pc1 = unpromoted_piece.at(pc1);
      pc2 = unpromoted_piece.at(pc2);

      int32_t pc1_t = piece_to_index.at(pc1);
      int32_t pc2_t = piece_to_index.at(pc2);

      if (pc1_t > pc2_t) {
        std::swap(pc1_t, pc2_t);
        std::swap(sq1, sq2);
      }

      int32_t key = (sq1 << 13) | (sq2 << 8) | (pc1_t << 4) | pc2_t;

      keys.push_back(key);
    }

  return keys;
}
}  // namespace Network
