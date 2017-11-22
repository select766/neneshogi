#include "../../extra/all.h"
#include "../../learn/learn.h"



#ifdef USER_ENGINE

#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#include <tiny_dnn/tiny_dnn.h>

const int dnn_channels = 61;
const int dnn_output_labels = 27 * (int)SQ_NB;

static tiny_dnn::network<tiny_dnn::sequential> *nn;
// USIに追加オプションを設定したいときは、この関数を定義すること。
// USI::init()のなかからコールバックされる。
void USI::extra_option(USI::OptionsMap & o)
{
    o["CNNPath"] << Option("");
}

// 起動時に呼び出される。時間のかからない探索関係の初期化処理はここに書くこと。
void Search::init()
{
}

// 定跡ファイル
Book::MemoryBook book;

// isreadyコマンドの応答中に呼び出される。時間のかかる処理はここに書くこと。
void  Search::clear()
{
    static bool first = true;
    if (first)
    {
        Book::read_book("book/standard_book.db", book);
        first = false;
    }

    nn = new tiny_dnn::network<tiny_dnn::sequential>();
    string cnn_path = Options["CNNPath"];
    if (cnn_path.length() > 0)
    {
        sync_cout << "info string Loading serialized nn" << sync_endl;
        nn->load(cnn_path);
        sync_cout << "info string nn load ok" << sync_endl;
    }
    else
    {
        sync_cout << "info string nn NOT loaded" << sync_endl;
    }
}

template<typename vec>
static void make_board(const Position &pos, vec &board)
{
    // channel 0~27: piece on board
    for (Square sq = SQ_11; sq < SQ_NB; sq++)
    {
        Piece pc = pos.piece_on(sq);
        int ch;
        if (pos.side_to_move() == BLACK)
        {
            if (pc >= W_PAWN)
            {
                ch = (int)pc - (int)W_PAWN + 14;
                board[ch * SQ_NB + sq] = 1;
            }
            else if (pc >= B_PAWN)
            {
                ch = (int)pc - (int)B_PAWN;
                board[ch * SQ_NB + sq] = 1;
            }
        }
        else
        {
            if (pc >= W_PAWN)
            {
                ch = (int)pc - (int)W_PAWN;
                board[ch * SQ_NB + Inv(sq)] = 1;
            }
            else if (pc >= B_PAWN)
            {
                ch = (int)pc - (int)B_PAWN + 14;
                board[ch * SQ_NB + Inv(sq)] = 1;
            }

        }
    }

    // channel 28~41: piece on board
    for (int phase = 0; phase < 2; phase++)
    {
        Color c = pos.side_to_move();
        int ch_ofs = 28 + phase * 7;
        if (phase > 0)
        {
            c = ~c;
        }

        for (Piece pr = PIECE_HAND_ZERO; pr < PIECE_HAND_NB; pr++)
        {
            Hand h = pos.hand_of(c);
            int n_hand = hand_count(h, pr);
            int ch = ch_ofs + pr - PIECE_HAND_ZERO;

            for (Square sq = SQ_11; sq < SQ_NB; sq++)
            {
                board[ch * SQ_NB + sq] = n_hand;
            }
        }
    }

    // channel 42~50: rank
    // channel 60: constant 1 on board
    for (Square sq = SQ_11; sq < SQ_NB; sq++)
    {
        Rank rank = rank_of(sq);
        File file = file_of(sq);
        Piece pc = pos.piece_on(sq);
        int ch;
        board[(42 + (int)rank) * SQ_NB + sq] = 1;
        board[(42 + 9 + (int)file) * SQ_NB + sq] = 1;
        board[(42 + 18) * SQ_NB + sq] = 1;
    }
}

static int get_move_index(Move move, Color side_to_move)
{
    Square sq_move_to = move_to(move);
    Square sq_move_to_rot = side_to_move == BLACK ? sq_move_to : Inv(sq_move_to);

    int ch;
    if (is_drop(move))
    {
        //駒打ち
        Piece dropped_piece = move_dropped_piece(move);
        ch = dropped_piece - PAWN + 20;
    }
    else
    {
        //駒打ち以外
        Square sq_move_from = move_from(move);
        Square sq_move_from_rot = side_to_move == BLACK ? sq_move_from : Inv(sq_move_from);
        File y_from = file_of(sq_move_from_rot);//筋 y座標
        Rank x_from = rank_of(sq_move_from_rot);//段 x座標
        File y_to = file_of(sq_move_to_rot);
        Rank x_to = rank_of(sq_move_to_rot);
        if (y_to == y_from)
        {
            if (x_to < x_from)
            {
                ch = 0;
            }
            else
            {
                ch = 1;
            }
        }
        else if (x_to == x_from)
        {
            if (y_to < y_from)
            {
                ch = 2;
            }
            else
            {
                ch = 3;
            }
        }
        else if ((y_to - y_from) == (x_to - x_from))
        {
            if (y_to < y_from)
            {
                ch = 4;
            }
            else
            {
                ch = 5;
            }
        }
        else if ((y_to - y_from) == (x_from - x_to))
        {
            if (y_to < y_from)
            {
                ch = 6;
            }
            else
            {
                ch = 7;
            }
        }
        else
        {
            if (y_to < y_from)
            {
                ch = 8;
            }
            else
            {
                ch = 9;
            }
        }

        if (is_promote(move))
        {
            ch += 10;
        }
    }

    return ch * SQ_NB + (int)sq_move_to_rot;
}

static float extract_move_score(tiny_dnn::vec_t &scores, Move move, Color side_to_move)
{
    return (float)scores[get_move_index(move, side_to_move)];
}

// 探索開始時に呼び出される。
// この関数内で初期化を終わらせ、slaveスレッドを起動してThread::search()を呼び出す。
// そのあとslaveスレッドを終了させ、ベストな指し手を返すこと。
void MainThread::think()
{
    static PRNG prng;

    Move best_move = MOVE_RESIGN;

    {
        auto it = book.find(rootPos);
        if (it != book.end()) {
            // 定跡にhitした。逆順で出力しないと将棋所だと逆順にならないという問題があるので逆順で出力する。
            const auto& move_list = it->second;
            for (auto it = move_list.rbegin(); it != move_list.rend(); it++)
                sync_cout << "info pv " << it->bestMove << " " << it->nextMove
                << " (" << fixed << setprecision(2) << (100 * it->prob) << "%)" // 採択確率
                << " score cp " << it->value << " depth " << it->depth << sync_endl;

            // このなかの一つをランダムに選択
            // 無難な指し手が選びたければ、採択回数が一番多い、最初の指し手(move_list[0])を選ぶべし。
            best_move = move_list[prng.rand(move_list.size())].bestMove;

            goto ID_END;
        }
    }

      // NNに盤面を与えて、方策スコアを得る
    {
        tiny_dnn::vec_t board;
        board.resize(dnn_channels * (int)SQ_NB);
        make_board(rootPos, board);
        tiny_dnn::vec_t scores = nn->predict(board);

        // 合法手生成
        MoveList<LEGAL> ml(rootPos);
        Color side_to_move = rootPos.side_to_move();
        float max_score = -10000;
        for (int i = 0; i < ml.size(); i++)
        {
            Move move = ml.at(i).move;
            float move_raw_score = extract_move_score(scores, move, side_to_move);
            if (move_raw_score > max_score)
            {
                max_score = move_raw_score;
                best_move = move;
            }
        }

        // softmax確率を出す(指し手生成自体には不要)
        float softmax_prob = 0.0F;
        {
            float softmax_sum = 0.0F;
            for (int i = 0; i < scores.size(); i++)
            {
                softmax_sum += expf(scores[i]);
            }
            softmax_prob = expf(max_score) / softmax_sum;
        }

        sync_cout << "info string probability " << softmax_prob << sync_endl;
    }
ID_END:;
    sync_cout << "bestmove " << best_move << sync_endl;
}

// 探索本体。並列化している場合、ここがslaveのエントリーポイント。
// MainThread::search()はvirtualになっていてthink()が呼び出されるので、MainThread::think()から
// この関数を呼び出したいときは、Thread::search()とすること。
void Thread::search()
{
}

void user_decode_train_sfen(Position& pos_, istringstream& is)
{
    std::string sfen_path;
    is >> sfen_path;
    long long offset, count;
    is >> offset;
    is >> count;
    
    _setmode(_fileno(stdout), _O_BINARY);

    fstream fs;
    fs.open(sfen_path, ios::in | ios::binary);

    fs.seekg(sizeof(Learner::PackedSfenValue) * offset);

    int block_size = 256;
    Learner::PackedSfenValue *sfen_buf = new Learner::PackedSfenValue[block_size];
    const int dnn_input_num_elements = dnn_channels * (int)SQ_NB;
    const int dnn_output_num_elements = 2;
    const int dnn_sample_size = dnn_input_num_elements * sizeof(float) + dnn_output_num_elements * sizeof(int);
    char* decoded_buf = new char[dnn_sample_size * block_size];
    while (count > 0)
    {
        int this_block_size = count > block_size ? block_size : count;
        fs.read((char*)sfen_buf, sizeof(Learner::PackedSfenValue) * this_block_size);
        char* decoded_buf_offset = decoded_buf;
        for (size_t i = 0; i < this_block_size; i++)
        {
            Position pos;
            pos.init();
            pos.set_from_packed_sfen(sfen_buf[i].sfen);
            float* vec_ptr = reinterpret_cast<float*>(decoded_buf_offset);
            memset(vec_ptr, 0, dnn_input_num_elements * sizeof(float));
            make_board<float*>(pos, vec_ptr);
            decoded_buf_offset += dnn_input_num_elements * sizeof(float);
            int* move_ptr = reinterpret_cast<int*>(decoded_buf_offset);
            move_ptr[0] = get_move_index((Move)sfen_buf[i].move, pos.side_to_move());//移動先
            move_ptr[1] = (int)sfen_buf[i].score;//評価値
            decoded_buf_offset += dnn_output_num_elements * sizeof(int);
        }

        std::cout.write(decoded_buf, dnn_sample_size * this_block_size);
        std::cout.flush();

        count -= this_block_size;
    }

    delete[] sfen_buf;
    delete[] decoded_buf;
}

#endif // USER_ENGINE

// USI拡張コマンド"user"が送られてくるとこの関数が呼び出される。実験に使ってください。
void user_test(Position& pos_, istringstream& is)
{
    std::string subcommand;
    is >> subcommand;
#ifdef USER_ENGINE
    if (subcommand == "decode_train_sfen")
    {
        user_decode_train_sfen(pos_, is);
    }
#endif
}
