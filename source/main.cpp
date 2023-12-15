#include <torch/script.h>
#include <torch/torch.h>

#include "args.h"
#include "network.h"
#include "search.h"
#include "usi.h"

int main(int argc, char *argv[]) {
  torch::InferenceMode guard(true);

  // --- コマンドライン引数の解析
  Args::parse(argc, argv);

  // --- QNNPACKの初期化
  bool use_quantized = false;
  for (auto e : torch::globalContext().supportedQEngines()) {
    if (e == torch::kQNNPACK || e == torch::kFBGEMM) {
      use_quantized = true;
      torch::globalContext().setQEngine(e);
    }
  }

  // --- ニューラルネットワークの読み込み
  torch::jit::script::Module model;
  if (use_quantized) {
    model = torch::jit::load("./model-sp-q8.pt");
    printf("[info] Set quantized mode\n");
  } else {
    model = torch::jit::load("./model-sp.pt");
    printf("[info] Set float mode\n");
  }

  model.eval();
  torch::jit::optimize_for_inference(model);
  Network::model = &model;
  printf("[info] Successfully loaded model\n");

  // --- 全体的な初期化
  Bitboards::init();
  Position::init();
  Search::init();

  // USIコマンドの応答部
  USI::loop(argc, argv);
}
