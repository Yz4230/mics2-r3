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

double evaluate(const Position &pos) {
  static torch::Tensor input = torch::zeros({1, 46800});

  double score = 0;

  // fill zero
  input.fill_(0);

  auto keys = Network::position_to_keys(pos);
  for (auto key : keys) {
    auto index = Network::key_to_index[key];
    input[0][index] = 1;
  }
  auto output = Network::model->forward({input}).toTensor();
  score = output[0][0].item<double>();

  // 手番側から見た評価値を返す
  return pos.side_to_move() == BLACK ? 1 - score : score;
}

}  // namespace Eval
