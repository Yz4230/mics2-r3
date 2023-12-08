#include <stdlib.h>
#include <string.h>

namespace Args {
int random_first_moves = 0;
int depth = 127;

void parse(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-r") == 0) {
      random_first_moves = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-d") == 0) {
      depth = atoi(argv[++i]);
    }
  }
}
}  // namespace Args
