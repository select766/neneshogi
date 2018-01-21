#pragma once
class UndoMoveInfo
{
public:
	// 盤上の駒2つ、持ち駒の数1つの以前の状態が記録できれば良い。
	// 駒の移動の時、移動元、移動先、持ち駒（駒を取った場合）が変化。
	// 駒打ちの時、移動先、持ち駒が変化。
	// 駒打ちなら移動先、移動元両方に同じ値を入れることにより場合分け不要とする。
	// 持ち駒は2*7で表されるが、1次元のアドレスで指定。
	// もう少し最適化できる気がする。
	int _from_sq, _to_sq, _hand_type;
	uint8_t _from_value, _to_value, _hand_value;

	UndoMoveInfo();
	UndoMoveInfo(int from_sq, uint8_t from_value, int to_sq, uint8_t to_value, int hand_type, uint8_t hand_value);
	~UndoMoveInfo();
};

