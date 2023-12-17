#include <torch/script.h>
#include <torch/torch.h>

#include "args.h"
#include "evaluate.h"
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
  torch::jit::script::Module model;
  if (use_quantized) {
    printf("[info] Set quantized mode\n");
    model = torch::jit::load("./model-cnn-q8.pt");
  } else {
    printf("[info] Set float mode\n");
    model = torch::jit::load("./model-cnn.pt");
  }

  model.eval();
  torch::jit::optimize_for_inference(model);
  Eval::model = &model;
  printf("[info] Successfully loaded model\n");

  // --- 全体的な初期化
  Bitboards::init();
  Position::init();
  Search::init();

  // USIコマンドの応答部
  USI::loop(argc, argv);
}
