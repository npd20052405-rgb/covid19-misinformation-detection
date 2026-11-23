"""Conservative text normalization; no learned state, no destructive raw edits."""
import re
import unicodedata


def normalize_text(text):
    if not isinstance(text, str):
        raise TypeError('Expected text string')
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().lower()


def self_check():
    assert normalize_text('  ＣOVID\u00a0 19  ') == 'covid 19'
    assert normalize_text('NOT safe! 12.5%') == 'not safe! 12.5%'
    assert normalize_text('https://t.co/ABC @USER') == 'https://t.co/abc @user'
    assert normalize_text(normalize_text(' A\nB ')) == 'a b'
    print('PASS: Unicode, whitespace, negation, numbers and idempotence checks')


if __name__ == '__main__':
    self_check()
