#include "evaluate.h"

#include "kp_index_map.h"
#include "network.h"

namespace Eval {
// clang-format off
constexpr int32_t piece_to_index[32] = {
    2147483647, 4, 2147483647, 2147483647, 3, 1, 0, 2,
    2147483647, 8, 2147483647, 2147483647, 7, 6, 5, 2147483647,
    2147483647, 4, 2147483647, 2147483647, 3, 1, 0, 2,
    2147483647, 8, 2147483647, 2147483647, 7, 6, 5, 2147483647,
};
// clang-format on

size_t position_to_kp(const Position &pos, uint16_t *kp) {
  size_t count = 0;

  for (auto king_color : {BLACK, WHITE}) {
    auto king_sq = pos.king_square(king_color);
    for (auto sq : SQ) {
      if (sq == king_sq) continue;

      auto piece = pos.piece_on(sq);
      if (piece == NO_PIECE || piece == B_KING || piece == W_KING) continue;

      bool is_opp_king = king_color != pos.side_to_move();
      bool is_opp_piece = color_of(piece) != pos.side_to_move();
      if (pos.side_to_move() == WHITE) {
        king_sq = Square(24 - king_sq);
        sq = Square(24 - sq);
      }
      int king_index = is_opp_king * 25 + king_sq;
      int piece_index = is_opp_piece * 225 + piece_to_index[piece] * 25 + sq;
      auto kp_index = kp_index_map[king_index][piece_index];
      kp[count++] = kp_index;
    }
  }

  return count;
}

void convert_position_to_input(const Position &pos, const torch::Tensor &dist) {
  static uint16_t kp_indices[20];

  // [21600]: 2駒関係(KP)の評価
  constexpr int kp_input_offset = 0;
  auto kp_count = position_to_kp(pos, kp_indices);
  for (size_t i = 0; i < kp_count; ++i) {
    dist[kp_input_offset + kp_indices[i]] = 1;
  }

  // [10]: 手駒の評価
  constexpr int hand_input_offset = kp_input_offset + 21600;
  for (auto color : {BLACK, WHITE}) {
    auto hand = pos.hand_of(color);
    for (auto piece : {PAWN, SILVER, BISHOP, ROOK, GOLD}) {
      int count = hand_count(hand, piece);
      int index = hand_input_offset + piece_to_index[piece];
      if (color == pos.side_to_move()) {
        dist[index] = count;
      } else {
        dist[index + 5] = count;
      }
    }
  }

  // [18]: 盤上の評価
  constexpr int board_input_offset = hand_input_offset + 10;
  for (auto sq : SQ) {
    auto piece = pos.piece_on(sq);
    if (piece == NO_PIECE || piece == B_KING || piece == W_KING) continue;

    int index = board_input_offset + piece_to_index[piece];
    if (color_of(piece) == pos.side_to_move()) {
      dist[index] += 1;
    } else {
      dist[index + 9] += 1;
    }
  }

  // [1]: 手番
  constexpr int turn_input_offset = board_input_offset + 18;
  dist[turn_input_offset] = pos.side_to_move() == BLACK;

  // [100]: 手数
  constexpr int moves_input_offset = turn_input_offset + 1;
  dist[moves_input_offset + pos.game_ply()] = 1;
}

double evaluate(const Position &pos) {
  torch::Tensor input = torch::zeros({1, 21729});

  convert_position_to_input(pos, input);
  auto output = Network::model->forward({input}).toTensor();

  // 　現在の手番が勝つ確率, [0, 1]
  auto score = output[0][0].item<double>();

  return score - 0.5;
}

}  // namespace Eval
