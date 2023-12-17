#include "evaluate.h"

namespace Eval {
torch::jit::script::Module *models[6] = {nullptr};

// clang-format off
constexpr uint8_t piece_c_to_py[32] = {
    0, 6, 255, 255, 5, 3, 2, 4, 1, 10, 255, 255, 9, 8, 7, 255,
    0, 6, 255, 255, 5, 3, 2, 4, 1, 10, 255, 255, 9, 8, 7, 255};
// clang-format on

void convert_position_to_input(const Position &pos, const torch::Tensor &dist) {
  //  POS_INDEX = np.arange(2*10*25).reshape(2, 10, 25) # 2: side, 10: piece,
  //  25: square HAND_INDEX = np.arange(2*5*2).reshape(2, 5*2) # 2: side, 5:
  //  kind of piece, 2: number of pieces PIECE_INDEX =
  //  np.arange(2*9*2).reshape(2, 9*2) # 2: side, 9: kind of piece, 2: number of
  //  pieces TURN_INDEX = np.arange(2) # 2: side PLY_INDEX = np.arange(100) #
  //  100: ply, (start from 0, over 99 is treated as 99)

  //  POS_INDEX.size, HAND_INDEX.size, PIECE_INDEX.size, TURN_INDEX.size,
  //  PLY_INDEX.size
  //  (500, 20, 36, 2, 100)

  constexpr int POS_START = 0;
  constexpr int POS_SIZE = 500;
  constexpr int HAND_START = POS_START + POS_SIZE;
  constexpr int HAND_SIZE = 20;
  constexpr int PIECE_START = HAND_START + HAND_SIZE;
  constexpr int PIECE_SIZE = 36;
  constexpr int TURN_START = PIECE_START + PIECE_SIZE;
  constexpr int TURN_SIZE = 2;
  constexpr int PLY_START = TURN_START + TURN_SIZE;
  constexpr int PLY_SIZE = 100;

  // [500]: 位置の評価
  // [36]: 盤上駒の評価
  bool pos_at_least_one[36] = {0};  // 0~17: own, 18~35: opp
  for (auto sq : SQ) {
    auto piece = pos.piece_on(sq);
    if (piece == NO_PIECE) continue;

    bool is_opp_piece = color_of(piece) != pos.side_to_move();
    int p = piece_c_to_py[piece];
    int pos_index = is_opp_piece * 250 + (p - 1) * 25 + sq;
    dist[POS_START + pos_index] = 1;

    if (piece == B_KING || piece == W_KING) continue;

    int piece_index = is_opp_piece * 18 + (p - 2) * 2;
    if (pos_at_least_one[piece_index]) {
      dist[PIECE_START + piece_index + 1] = 1;
    } else {
      dist[PIECE_START + piece_index] = 1;
      pos_at_least_one[piece_index] = true;
    }
  }

  // [20]: 持ち駒の評価
  bool hand_at_least_one[20] = {0};  // 0~9: own, 10~19: opp
  for (auto color : {BLACK, WHITE}) {
    auto hand = pos.hand_of(color);
    bool is_opp_hand = color != pos.side_to_move();
    for (auto piece : {PAWN, SILVER, BISHOP, ROOK, GOLD}) {
      int count = hand_count(hand, piece);
      if (count == 0) continue;

      int p = piece_c_to_py[piece];
      int hand_index = is_opp_hand * 10 + (p - 2) * 2;
      if (hand_at_least_one[hand_index]) {
        dist[HAND_START + hand_index + 1] = 1;
      } else {
        dist[HAND_START + hand_index] = 1;
        hand_at_least_one[hand_index] = true;
      }
    }
  }

  // [2]: 手番
  dist[TURN_START + pos.side_to_move()] = 1;

  // [100]: 手数
  int ply = std::min(pos.game_ply(), 100);
  dist[PLY_START + ply - 1] = 1;  // game_ply()は1から始まる
}

double evaluate(const Position &pos) {
  const auto model = Eval::models[std::min(pos.game_ply() / 10, 5)];
  torch::Tensor input = torch::zeros({1, 658});

  input.fill_(0);
  convert_position_to_input(pos, input[0]);
  torch::save(input, "input.pt");
  auto output = model->forward({input}).toTensor();

  // 　現在の手番が勝つ確率, [0, 1]
  auto score = output[0][0].item<double>();

  return score * 2 - 1;  // [-1, 1]
}

}  // namespace Eval
