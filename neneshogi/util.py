import string


def strip_path(path: str) -> str:
    """
    ファイルパスの両端から不要な文字を除去する。
    Windowsで便利なように、両端から",'を除去する。
    :param path:
    :return:
    """
    # 「パスのコピー」でダブルクオーテーションが入るのでそれをそのまま利用したい
    return path.strip(string.whitespace + "'\"")
