#include "search.h"

#include <algorithm>
#include <map>
#include <random>
#include <thread>

#include "args.h"
#include "evaluate.h"
#include "misc.h"
#include "usi.h"

struct MovePicker {
  // 静止探索で使用
  MovePicker(const Position &pos_, Square recapSq) : pos(pos_) {
    if (pos.in_check())
      endMoves = generateMoves<EVASIONS>(pos, currentMoves);
    else
      endMoves = generateMoves<RECAPTURES>(pos, currentMoves, recapSq);
  }

  Move nextMove() {
    if (currentMoves == endMoves) return MOVE_NONE;
    return *currentMoves++;
  }

 private:
  const Position &pos;
  ExtMove moves[MAX_MOVES], *currentMoves = moves, *endMoves = moves;
};

namespace Search {
// 探索開始局面で思考対象とする指し手の集合。
RootMoves rootMoves;

// 持ち時間設定など。
LimitsType Limits;

// 今回のgoコマンドでの探索ノード数。
uint64_t Nodes;

// 探索中にこれがtrueになったら探索を即座に終了すること。
bool Stop;

}  // namespace Search

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init() {}

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void Search::clear() {}

Value search(Position &pos, Value alpha, Value beta, int depth,
             int ply_from_root);

// 探索を開始する
void Search::start_thinking(const Position &rootPos, StateListPtr &states,
                            LimitsType limits) {
  Limits = limits;
  rootMoves.clear();
  Nodes = 0;
  Stop = false;

  for (Move move : MoveList<LEGAL>(rootPos)) rootMoves.emplace_back(move);
  ASSERT_LV3(states.get());

  Position *pos_ptr = const_cast<Position *>(&rootPos);
  search(*pos_ptr);
}

// 探索本体
void Search::search(Position &pos) {
  // 探索で返す指し手
  Move bestMove = MOVE_RESIGN;

  if (rootMoves.size() == 0) {
    // 合法手が存在しない
    Stop = true;
    goto END;
  }

  /* ここから探索部を記述する */
  {
    std::random_device rnd;
    std::mt19937 mt(rnd());
    int i = mt() % rootMoves.size();
    bestMove = rootMoves[i].pv[0];
  }
  /* 探索部ここまで */

END:;
  std::cout << "bestmove " << bestMove << std::endl;
}

// アルファ・ベータ法(apha-beta pruning)
Value search(Position &pos, Value alpha, Value beta, int depth,
             int ply_from_root) {
  // 末端では評価関数を呼び出す
  if (depth <= 0) return Eval::evaluate(pos);

  // 初期値はマイナス∞
  Value maxValue = -VALUE_INFINITE;
  // do_move() に必要
  StateInfo si;
  // この局面で do_move() された合法手の数
  int moveCount = 0;

  // 千日手の検出
  RepetitionState draw_type = pos.is_repetition();
  if (draw_type != REPETITION_NONE)
    return draw_value(draw_type, pos.side_to_move());

  for (ExtMove m : MoveList<LEGAL>(pos)) {
    // 局面を 1 手進める
    pos.do_move(m.move, si);
    ++moveCount;
    // 再帰的に search() を呼び出す. このとき, 評価値にマイナスをかける
    Value value = -search(pos, -beta, -alpha, depth - 1, ply_from_root + 1);
    // 局面を 1 手戻す
    pos.undo_move(m.move);

    // 探索の終了
    if (Search::Stop) return VALUE_ZERO;

    // 局面評価値の更新
    if (value > maxValue) maxValue = value;

    // アルファ値の更新
    if (value > alpha) alpha = value;

    // beta cut
    if (alpha >= beta) break;
  }

  // 合法手の数が 0 のとき詰んでいる
  // 詰みのスコアを返す
  if (moveCount == 0) return mated_in(ply_from_root);

  // 最も良い評価値を返す
  return maxValue;
}
