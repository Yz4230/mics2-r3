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
  for (Square sq : SQ) score += PieceValue[pos.piece_on(sq)];

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
