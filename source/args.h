namespace Args {
// 初手から何手までをランダムで進めるか
// arg: -r
extern int random_first_moves;

// 探索する深さ
// arg: -d
extern int depth;

// 引数の解析
void parse(int argc, char *argv[]);
}  // namespace Args
