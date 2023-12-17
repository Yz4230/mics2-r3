#include "evaluate.h"

namespace Eval {
torch::jit::script::Module *model = nullptr;

// clang-format off
constexpr uint8_t piece_c_to_py[32] = {
    0, 6, 255, 255, 5, 3, 2, 4, 1, 10, 255, 255, 9, 8, 7, 255,
    0, 6, 255, 255, 5, 3, 2, 4, 1, 10, 255, 255, 9, 8, 7, 255};
// clang-format on

void convert_position_to_input(const Position &pos, const torch::Tensor &dist) {
  constexpr uint8_t SQ_TO_XY[25][2] = {
      {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4},  // 0~4
      {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4},  // 5~9
      {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4},  // 10~14
      {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4},  // 15~19
      {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}   // 20~24
  };

  constexpr uint8_t POS_INDEX[2][10] = {
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
  };

  constexpr uint8_t HAND_INDEX[2][5][2] = {
      {{20, 21}, {22, 23}, {24, 25}, {26, 27}, {28, 29}},
      {{30, 31}, {32, 33}, {34, 35}, {36, 37}, {38, 39}}};

  // clang-format off
  constexpr uint8_t PIECE_INDEX[2][9][2] = {
    {{40, 41}, {42, 43}, {44, 45}, {46, 47}, {48, 49}, {50, 51}, {52, 53}, {54, 55}, {56, 57}},
    {{58, 59}, {60, 61}, {62, 63}, {64, 65}, {66, 67}, {68, 69}, {70, 71}, {72, 73}, {74, 75}}};
  // clang-format on

  constexpr uint8_t TURN_INDEX[2] = {76, 77};

  // [20]: 位置の評価
  // [36]: 盤上駒の評価
  bool pos_at_least_one[36] = {0};  // 0~17: own, 18~35: opp
  for (auto sq : SQ) {
    auto piece = pos.piece_on(sq);
    if (piece == NO_PIECE) continue;

    bool is_opp_piece = color_of(piece) != pos.side_to_move();
    int p = piece_c_to_py[piece];
    int pos_index = POS_INDEX[is_opp_piece][p - 1];
    int px = SQ_TO_XY[sq][0];
    int py = SQ_TO_XY[sq][1];
    dist[pos_index][px][py] = 1;

    if (piece == B_KING || piece == W_KING) continue;

    int piece_index = PIECE_INDEX[is_opp_piece][p - 2][0];
    if (pos_at_least_one[piece_index]) {
      dist[piece_index + 1].fill_(1);
    } else {
      dist[piece_index].fill_(1);
      pos_at_least_one[piece_index] = true;
    }
  }

  // [20]: 持ち駒の評価
  for (auto color : {BLACK, WHITE}) {
    auto hand = pos.hand_of(color);
    bool is_opp_hand = color != pos.side_to_move();
    for (auto piece : {PAWN, SILVER, BISHOP, ROOK, GOLD}) {
      int count = hand_count(hand, piece);
      if (count == 0) continue;

      int p = piece_c_to_py[piece];
      int hand_index = HAND_INDEX[is_opp_hand][p - 2][0];
      dist[hand_index].fill_(1);
      if (count == 2) {
        dist[hand_index + 1].fill_(1);
      }
    }
  }

  // [2]: 手番
  dist[TURN_INDEX[pos.side_to_move()]].fill_(1);
}

double evaluate(const Position &pos) {
  torch::Tensor input = torch::zeros({1, 78, 5, 5});

  input.fill_(0);
  convert_position_to_input(pos, input[0]);
  torch::save(input, "input.pt");
  auto output = model->forward({input}).toTensor();

  // 　現在の手番が勝つ確率, [0, 1]
  auto score = output[0][0].item<double>();

  return score * 2 - 1;  // [-1, 1]
}

}  // namespace Eval
