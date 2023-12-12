#include <stdio.h>

#include "../evaluate.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"

void user_test(Position& pos, std::istringstream& is) {
  double score = Eval::evaluate(pos);
  printf("score = %f\n", score);
}
