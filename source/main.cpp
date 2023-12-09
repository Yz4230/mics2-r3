#include "args.h"
#include "network.h"
#include "search.h"
#include "usi.h"

int main(int argc, char *argv[]) {
  // --- コマンドライン引数の解析
  Args::parse(argc, argv);

  // weightファイルの読み込み
  auto network = Network::LinearNetwork::from_weights_file("weights.txt");
  Network::network = &network;

  // key_to_indexファイルの読み込み
  auto key_to_index = Network::from_key_to_index_file("key_to_index.txt");
  Network::key_to_index = key_to_index;

  // --- 全体的な初期化
  Bitboards::init();
  Position::init();
  Search::init();

  // USIコマンドの応答部
  USI::loop(argc, argv);
}
