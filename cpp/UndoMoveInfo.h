#pragma once
class UndoMoveInfo
{
public:
	// �Տ�̋�2�A������̐�1�̈ȑO�̏�Ԃ��L�^�ł���Ηǂ��B
	// ��̈ړ��̎��A�ړ����A�ړ���A������i���������ꍇ�j���ω��B
	// ��ł��̎��A�ړ���A������ω��B
	// ��ł��Ȃ�ړ���A�ړ��������ɓ����l�����邱�Ƃɂ��ꍇ�����s�v�Ƃ���B
	// �������2*7�ŕ\����邪�A1�����̃A�h���X�Ŏw��B
	// ���������œK���ł���C������B
	int _from_sq, _to_sq, _hand_type;
	uint8_t _from_value, _to_value, _hand_value;

	UndoMoveInfo();
	UndoMoveInfo(int from_sq, uint8_t from_value, int to_sq, uint8_t to_value, int hand_type, uint8_t hand_value);
	~UndoMoveInfo();
};

