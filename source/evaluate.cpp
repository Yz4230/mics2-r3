#include "evaluate.h"

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

Value evaluate(const Position &pos) {
  auto score = VALUE_ZERO;

  // 局面の評価
  for (Square sq : SQ) {
    // より相手陣に進んでいる駒ほど評価値を高くする
    const auto pc = pos.piece_on(sq);
    if (pc == NO_PIECE) continue;

    const auto c = color_of(pc);
    double r = (double)rank_of(sq);
    if (c == BLACK) {
      // 先手から見た段数に変換
      r = RANK_5 - r;
    }
    r = (r * 0.05) + 1;

    if (pc == B_KING || pc == W_KING) {
      r = 1;
    }
    score += PieceValue[pos.piece_on(sq)] * r;
  }

  // 手駒の評価
  for (Color c : COLOR) {
    auto hand = pos.hand_of(c);
    for (Piece pc : {PAWN, SILVER, BISHOP, ROOK, GOLD}) {
      auto s = Value(hand_count(hand, pc) * PieceValue[pc]);
      score += (c == BLACK ? s : -s);
    }
  }

  // 手番側から見た評価値を返す
  return pos.side_to_move() == BLACK ? score : -score;
}

}  // namespace Eval
