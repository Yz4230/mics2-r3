#ifndef _EVALUATE_H_
#define _EVALUATE_H_

#include "position.h"
#include "types.h"

namespace Eval {
// Apery(WCSC26)の駒割り
enum {
  // 駒の評価値
  PawnValue = 100,
  SilverValue = 640,
  GoldValue = 690,
  BishopValue = 890,
  RookValue = 1040,
  ProPawnValue = 420,
  ProSilverValue = 670,
  HorseValue = 1150,
  DragonValue = 1300,
  KingValue = 15000,
};

// 駒の価値のテーブル(後手の駒は負の値)
extern int PieceValue[PIECE_NB];

Value evaluate(const Position &pos);
} // namespace Eval

#endif
