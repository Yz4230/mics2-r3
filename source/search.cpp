#include "search.h"

#include <algorithm>
#include <random>
#include <thread>

#include "args.h"
#include "evaluate.h"
#include "misc.h"
#include "network.h"
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
    /* 時間制御 */
    Color us = pos.side_to_move();
    std::thread *timerThread = nullptr;

    // 今回は秒読み以外の設定は考慮しない
    s64 endTime = Limits.byoyomi[us] - 150;

    timerThread = new std::thread([&] {
      while (Time.elapsed() < endTime && !Stop)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      Stop = true;
    });

    StateInfo state;

    auto cmp = [](const RootMove &a, const RootMove &b) {
      return a.score > b.score;
    };

    if (pos.game_ply() <= Args::random_first_moves) {
      // {random_first_moves}手目まではランダムに指す
      std::random_device rnd;
      std::mt19937 mt(rnd());
      int i = mt() % rootMoves.size();
      bestMove = rootMoves[i].pv[0];
    } else {
      // 以降は最善手を選択
      // 反復深化
      for (int depth = 3; depth <= Args::depth && !Stop; ++depth) {
        for (size_t i = 0; i < rootMoves.size() && !Stop; ++i) {
          // 合法手のi番目から探索を開始
          Move move = rootMoves[i].pv[0];
          // 局面を1手進める
          pos.do_move(move, state);
          // 再帰的に探索(アルファ・ベータ法)
          double value =
              -search(pos, -VALUE_INFINITE, VALUE_INFINITE, depth - 1, 0);
          rootMoves[i].score = value;
          // 局面を1手戻す
          pos.undo_move(move);
        }
        std::stable_sort(rootMoves.begin(), rootMoves.end(), cmp);
        bestMove = rootMoves[0].pv[0];
        std::cout << USI::pv(pos, depth) << std::endl;
      }
    }

    // タイマースレッド終了
    Stop = true;
    if (timerThread != nullptr) {
      timerThread->join();
      delete timerThread;
    }
  }
  /* 探索部ここまで */

END:;
  std::cout << "bestmove " << bestMove << std::endl;
}

// アルファ・ベータ法(alpha-beta pruning)
double Search::search(Position &pos, double alpha, double beta, int depth,
                      int ply_from_root) {
  constexpr int INPUT_ALLOC_SIZE = 8;
  static torch::Tensor input = torch::zeros({INPUT_ALLOC_SIZE, 658});
  static std::vector<c10::IValue> input_container{input};

  // 初期値はマイナス∞
  double maxValue = -VALUE_INFINITE;
  // do_move() に必要
  StateInfo si;
  // この局面で do_move() された合法手の数
  int moveCount = 0;

  // 千日手の検出
  RepetitionState draw_type = pos.is_repetition();
  if (draw_type != REPETITION_NONE)
    return draw_value(draw_type, pos.side_to_move());

  if (depth > 1) {
    for (ExtMove m : MoveList<LEGAL>(pos)) {
      // 局面を 1 手進める
      pos.do_move(m.move, si);
      ++moveCount;

      double value = -search(pos, -beta, -alpha, depth - 1, ply_from_root + 1);

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
  } else {
    int count = 0;
    auto move_list = MoveList<LEGAL>(pos);
    size_t move_list_size = move_list.size();
    size_t trailing_start_index =
        (move_list_size / INPUT_ALLOC_SIZE) * INPUT_ALLOC_SIZE;

    // 末端の評価を並列で行う
    for (size_t i = 0; i < move_list_size; ++i) {
      ExtMove m = move_list.at(i);
      // 局面を 1 手進める
      pos.do_move(m.move, si);
      ++moveCount;

      // 局面を入力に変換
      Eval::convert_position_to_input(pos, input[count]);
      count++;

      if (count == INPUT_ALLOC_SIZE || i >= trailing_start_index) {
        auto output = Network::model->forward(input_container).toTensor();
        input.fill_(0);  // 使い回すので初期化

        for (int j = 0; j < count; ++j) {
          auto value = -(output[j][0].item<double>() * 2 - 1);
          if (value > maxValue) maxValue = value;
        }
        count = 0;

        // アルファ値の更新
        if (maxValue > alpha) alpha = maxValue;
      }
      // 局面を 1 手戻す
      pos.undo_move(m.move);

      // 探索の終了
      if (Search::Stop) return VALUE_ZERO;

      // beta cut
      if (alpha >= beta) break;
    }
  }

  // 合法手の数が 0 のとき詰んでいる
  // 詰みのスコアを返す
  if (moveCount == 0) return mated_in(ply_from_root);

  // 最も良い評価値を返す
  return maxValue;
}
