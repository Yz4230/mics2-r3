#include <torch/script.h>

#include "args.h"
#include "network.h"
#include "search.h"
#include "usi.h"

int main(int argc, char *argv[]) {
  c10::InferenceMode guard(true);

  // --- コマンドライン引数の解析
  Args::parse(argc, argv);

  // --- QNNPACKの初期化
  bool use_quantized = false;
  for (auto e : at::globalContext().supportedQEngines()) {
    if (e == c10::QEngine::QNNPACK || e == c10::QEngine::X86) {
      use_quantized = true;
      at::globalContext().setQEngine(e);
      break;
    }
  }

  // --- ニューラルネットワークの読み込み
  torch::jit::script::Module model;
  if (use_quantized) {
    model = torch::jit::load("./model-kp-q8.pt");
    printf("[info] Set quantized mode\n");
  } else {
    model = torch::jit::load("./model-kp.pt");
    printf("[info] Set float mode\n");
  }

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
