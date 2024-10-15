from text.symbols import *

symbol_to_id = {s: i for i, s in enumerate(symbols)}


def convert_to_ids(phonemes, tones, language):
    # 转换为ids
    # 将一段清理过的文本（cleaned_text）转换为一系列与文本中符号对应的ID
    """
    Converts a string of phone symbols to a sequence of IDs corresponding to the symbols in the text.
    Args:
      phonemes (str): A string of phone symbols to convert to a sequence of IDs.
      tones (list of int): A list of tone values corresponding to each phone symbol.
      language (str): The language identifier which determines the tone offset and language ID.

    Returns:
      tuple: A tuple containing:
        - phones_ids (list of int): A list of integers corresponding to the phone symbols.
        - tones_list (list of int): A list of adjusted tone values.
        - lang_ids (list of int): A list of language IDs corresponding to each phone symbol.
    """
    # 输入的 phones, 一个字符串列表，代表音素。
    phones_ids = [symbol_to_id[symbol] for symbol in phonemes]
    tone_start = language_tone_start_map[language]  # {'zh': 0, 'ja': 6, 'en': 8}
    tones_list = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones_ids]
    # phones（符号ID列表），tones（调整后的音调列表），lang_ids（语言ID列表）
    return phones_ids, tones_list, lang_ids


def get_bert(norm_text, word2ph, language, device, style_text=None, style_weight=0.7):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    lang_bert_func_map = {"zh": zh_bert, "en": en_bert, "ja": jp_bert}
    bert = lang_bert_func_map[language](
        norm_text, word2ph, device, style_text, style_weight
    )
    return bert


def check_bert_models():
    import json
    from pathlib import Path

    from config import config
    from utils.bert_utils import check_bert

    if config.mirror.lower() == "openi":
        import openi

        kwargs = {"token": config.openi_token} if config.openi_token else {}
        openi.login(**kwargs)

    with open("./bert/bert_models.json", "r") as fp:
        models = json.load(fp)
        for k, v in models.items():
            local_path = Path("./bert").joinpath(k)
            check_bert(v["repo_id"], v["files"], local_path)


def init_openjtalk():
    import platform

    if platform.platform() == "Linux":
        import pyopenjtalk

        pyopenjtalk.g2p("こんにちは，世界。")


init_openjtalk()
check_bert_models()
