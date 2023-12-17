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

  constexpr int num_models = 6;
  // --- ニューラルネットワークの読み込み
  torch::jit::script::Module models[num_models];
  if (use_quantized) {
    printf("[info] Set quantized mode\n");
    // model = torch::jit::load("./model-sp-q8.pt");
    models[0] = torch::jit::load("./model-sp-4_10-q8.pt");
    models[1] = torch::jit::load("./model-sp-10_20-q8.pt");
    models[2] = torch::jit::load("./model-sp-20_30-q8.pt");
    models[3] = torch::jit::load("./model-sp-30_40-q8.pt");
    models[4] = torch::jit::load("./model-sp-40_50-q8.pt");
    models[5] = torch::jit::load("./model-sp-50_99-q8.pt");
  } else {
    printf("[info] Set float mode\n");
    // model = torch::jit::load("./model-sp.pt");
    throw std::runtime_error("float mode is not supported");
  }

  for (int i = 0; i < num_models; i++) {
    models[i].eval();
    torch::jit::optimize_for_inference(models[i]);
    Eval::models[i] = &models[i];
  }

  printf("[info] Successfully loaded model\n");

  // --- 全体的な初期化
  Bitboards::init();
  Position::init();
  Search::init();

  // USIコマンドの応答部
  USI::loop(argc, argv);
}
