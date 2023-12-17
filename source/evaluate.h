#ifndef _EVALUATE_H_
#define _EVALUATE_H_

#include <torch/torch.h>

#include "position.h"
#include "types.h"

namespace Eval {
extern torch::jit::script::Module *models[6];

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

void convert_position_to_input(const Position &pos, const torch::Tensor &dist);

double evaluate(const Position &pos);
}  // namespace Eval

#endif
