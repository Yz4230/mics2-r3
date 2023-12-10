#include <torch/script.h>

#include "args.h"
#include "network.h"
#include "search.h"
#include "usi.h"

int main(int argc, char *argv[]) {
  // --- コマンドライン引数の解析
  Args::parse(argc, argv);

  // --- ニューラルネットワークの読み込み
  torch::jit::script::Module model = torch::jit::load("./model-hbtm.pt");
  model.eval();
  Network::model = &model;
  printf("[info] Successfully loaded model\n");

  // --- 全体的な初期化
  Bitboards::init();
  Position::init();
  Search::init();

  // USIコマンドの応答部
  USI::loop(argc, argv);
}
