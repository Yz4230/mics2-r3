#include "args.h"
#include "search.h"
#include "usi.h"

int main(int argc, char *argv[]) {
  // --- コマンドライン引数の解析
  Args::parse(argc, argv);

  // --- 全体的な初期化
  Bitboards::init();
  Position::init();
  Search::init();

  // USIコマンドの応答部
  USI::loop(argc, argv);
}
