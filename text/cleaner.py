from text import chinese, japanese, english, convert_to_ids


language_module_map = {"zh": chinese, "ja": japanese, "en": english}


def text_to_phonemes(text, language):
    language_module = language_module_map[language]
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
