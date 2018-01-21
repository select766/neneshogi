#include "stdafx.h"
#include "UndoMoveInfo.h"


UndoMoveInfo::UndoMoveInfo()
{
}

UndoMoveInfo::UndoMoveInfo(int from_sq, uint8_t from_value, int to_sq, uint8_t to_value, int hand_type, uint8_t hand_value)
	: _from_sq(from_sq), _from_value(from_value), _to_sq(to_sq), _to_value(to_value), _hand_type(hand_type), _hand_value(hand_value)
{
}


UndoMoveInfo::~UndoMoveInfo()
{
}
