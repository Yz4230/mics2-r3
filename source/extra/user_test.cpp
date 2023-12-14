#include <stdio.h>

#include <algorithm>
#include <list>
#include <map>

#include "../evaluate.h"
#include "../search.h"
#include "../usi.h"

void user_test(Position& pos, std::istringstream& is) {
  Search::Stop = false;
  Search::RootMoves rootMoves;
  auto cmp = [](const Search::RootMove& a, const Search::RootMove& b) {
    return a.score > b.score;
  };

  for (Move move : MoveList<LEGAL>(pos)) rootMoves.emplace_back(move);

  StateInfo state;
  for (int depth = 3; depth <= 64; ++depth) {
    for (auto& rootMove : rootMoves) {
      // 合法手のi番目から探索を開始
      Move move = rootMove.pv[0];
      // 局面を1手進める
      pos.do_move(move, state);

      // 再帰的に探索(アルファ・ベータ法)
      double value =
          -Search::search(pos, -VALUE_INFINITE, VALUE_INFINITE, depth - 1, 0);

      rootMove.score = value;

      // 局面を1手戻す
      pos.undo_move(move);
    }
    std::stable_sort(rootMoves.begin(), rootMoves.end(), cmp);
    Move bestMove = rootMoves[0].pv[0];
    printf("depth: %d score: %f bestmove: %s\n", depth, rootMoves[0].score,
           USI::move(bestMove).c_str());
  }

  Search::Stop = true;
}
