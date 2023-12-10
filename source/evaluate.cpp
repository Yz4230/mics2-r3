#include "evaluate.h"

#include "network.h"

namespace Eval {
int PieceValue[PIECE_NB] = {
    // 先手の駒の価値
    0,
    PawnValue,
    0,
    0,
    SilverValue,
    BishopValue,
    RookValue,
    GoldValue,
    KingValue,
    ProPawnValue,
    0,
    0,
    ProSilverValue,
    HorseValue,
    DragonValue,
    0,

    // 後手の駒の価値は負の値にする
    0,
    -PawnValue,
    0,
    0,
    -SilverValue,
    -BishopValue,
    -RookValue,
    -GoldValue,
    -KingValue,
    -ProPawnValue,
    0,
    0,
    -ProSilverValue,
    -HorseValue,
    -DragonValue,
    0,
};

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

double evaluate(const Position &pos) {
  static torch::Tensor input = torch::zeros({1, 46828});
  static const std::unordered_map<Piece, int32_t> piece_to_index = {
      {Piece::B_ROOK, 0},   {Piece::B_BISHOP, 1},     {Piece::B_GOLD, 2},
      {Piece::B_SILVER, 3}, {Piece::B_PAWN, 4},       {Piece::B_DRAGON, 5},
      {Piece::B_HORSE, 6},  {Piece::B_PRO_SILVER, 7}, {Piece::B_PRO_PAWN, 8},
      {Piece::W_ROOK, 0},   {Piece::W_BISHOP, 1},     {Piece::W_GOLD, 2},
      {Piece::W_SILVER, 3}, {Piece::W_PAWN, 4},       {Piece::W_DRAGON, 5},
      {Piece::W_HORSE, 6},  {Piece::W_PRO_SILVER, 7}, {Piece::W_PRO_PAWN, 8},
  };

  double score = 0;

  // fill zero
  input.fill_(0);

  // 0~46799: 2駒関係の評価
  auto keys = position_to_keys(pos);
  for (auto key : keys) {
    auto index = Network::key_to_index[key];
    input[0][index] = 1;
  }

  // 46800~46809: 手駒の評価
  for (auto color : {BLACK, WHITE}) {
    auto hand = pos.hand_of(color);
    for (auto piece : {PAWN, SILVER, BISHOP, ROOK, GOLD}) {
      int count = hand_count(hand, piece);
      int index = 46800 + piece_to_index.at(piece);
      if (color == BLACK) {
        input[0][index] = count > 0;
      } else {
        input[0][index + 5] = count > 0;
      }
    }
  }

  // 46810~46819: 盤上の駒の価値
  for (auto sq : SQ) {
    auto piece = pos.piece_on(sq);
    if (piece == NO_PIECE || piece == B_KING || piece == W_KING) continue;

    int index = 46810 + piece_to_index.at(piece);
    if (pos.side_to_move() == BLACK) {
      input[0][index] += 1;
    } else {
      input[0][index + 5] += 1;
    }
  }

  auto output = Network::model->forward({input}).toTensor();
  score = output[0][0].item<double>();

  // 手番側から見た評価値を返す
  return pos.side_to_move() == BLACK ? score : 1 - score;
}

}  // namespace Eval
