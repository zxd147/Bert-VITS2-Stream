from . import chinese, japanese, convert_to_ids
from .fix import japanese as japanese_fix


language_module_map = {"ZH": chinese, "JP": japanese}
language_module_map_fix = {"ZH": chinese, "JP": japanese_fix}


def text_to_phonemes(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_fix(text, language):
    """使用dev分支修复"""
    language_module = language_module_map_fix[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = text_to_phonemes(text, language)
    return convert_to_ids(phones, tones, language)


if __name__ == "__main__":
    pass
