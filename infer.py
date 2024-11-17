
import torch

import commons
from utils.model_utils import load_checkpoint
from models import SynthesizerTrn
from text import convert_to_ids, get_bert
from text.cleaner import text_to_phonemes
from text.symbols import symbols


def get_model(model_path: str, device: str, hps):
    # 当前版本模型 model
    model = SynthesizerTrn(len(symbols),
                           hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           n_speakers=hps.data.n_speakers,
                           **hps.model).to(device)
    model.eval()
    _ = load_checkpoint(model_path, model, None, skip_optimizer=True)
    return model


def get_all_bert(text, language, hps, device, style_text=None, style_weight=0.7):
    # 在此处实现当前版本的get_all_bert
    style_text = None if style_text == "" else style_text
    norm_text, phonemes, tones, word2ph = text_to_phonemes(text, language)
    phone, tone, lang = convert_to_ids(phonemes, tones, language)  # 序列
    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        lang = commons.intersperse(lang, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = get_bert(norm_text, word2ph, language, device, style_text, style_weight)
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone
    if language == "zh":
        bert = bert_ori
        ja_bert = torch.randn(1024, len(phone))
        en_bert = torch.randn(1024, len(phone))
    elif language == "ja":
        bert = torch.randn(1024, len(phone))
        ja_bert = bert_ori
        en_bert = torch.randn(1024, len(phone))
    elif language == "en":
        bert = torch.randn(1024, len(phone))
        ja_bert = torch.randn(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language should be zh, ja or en")
    assert bert.shape[-1] == len(phone), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    lang = torch.LongTensor(lang)
    return bert, ja_bert, en_bert, phone, tone, lang


def infer(text, speaker, language, hps, model, device, sdp_ratio, noise_scale, noise_scale_w, length_scale,
          skip_start=False, skip_end=False, style_text=None, style_weight=0.7):
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_all_bert(text, language, hps, device,
                                                                   style_text=style_text, style_weight=style_weight)
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        sid = torch.LongTensor([hps.data.spk2id[speaker]]).to(device)
        audio = \
            model.infer(x_tst, x_tst_lengths, sid, tones, lang_ids, bert, ja_bert, en_bert, sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
                0, 0].data.cpu().float().numpy()
    del (phones, x_tst, tones, lang_ids, bert, x_tst_lengths, sid, ja_bert, en_bert)  # , emo
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return audio


def infer_multilang(text, speaker, language, hps, model, device, sdp_ratio, noise_scale, noise_scale_w, length_scale,
                    skip_start=False, skip_end=False, style_text=None, style_weight=0.7):
    bert, ja_bert, en_bert, phones, tones, lang_ids = [], [], [], [], [], []
    for idx, (txt, lang) in enumerate(zip(text, language)):
        skip_start = (idx != 0) or (skip_start and idx == 0)
        skip_end = (idx != len(language) - 1) or skip_end
        (temp_bert, temp_ja_bert, temp_en_bert, temp_phones,
         temp_tones, temp_lang_ids) = get_all_bert(txt, lang, hps, device,
                                                   style_text=style_text, style_weight=style_weight)
        if skip_start:
            temp_bert = temp_bert[:, 3:]
            temp_ja_bert = temp_ja_bert[:, 3:]
            temp_en_bert = temp_en_bert[:, 3:]
            temp_phones = temp_phones[3:]
            temp_tones = temp_tones[3:]
            temp_lang_ids = temp_lang_ids[3:]
        if skip_end:
            temp_bert = temp_bert[:, :-2]
            temp_ja_bert = temp_ja_bert[:, :-2]
            temp_en_bert = temp_en_bert[:, :-2]
            temp_phones = temp_phones[:-2]
            temp_tones = temp_tones[:-2]
            temp_lang_ids = temp_lang_ids[:-2]
        bert.append(temp_bert)
        ja_bert.append(temp_ja_bert)
        en_bert.append(temp_en_bert)
        phones.append(temp_phones)
        tones.append(temp_tones)
        lang_ids.append(temp_lang_ids)
    bert = torch.concatenate(bert, dim=1)
    ja_bert = torch.concatenate(ja_bert, dim=1)
    en_bert = torch.concatenate(en_bert, dim=1)
    phones = torch.concatenate(phones, dim=0)
    tones = torch.concatenate(tones, dim=0)
    lang_ids = torch.concatenate(lang_ids, dim=0)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        sid = torch.LongTensor([hps.data.spk2id[speaker]]).to(device)
        audio = \
            model.infer(x_tst, x_tst_lengths, sid, tones, lang_ids, bert, ja_bert, en_bert, sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
                0, 0].data.cpu().float().numpy()
        del (phones, x_tst, tones, lang_ids, bert, x_tst_lengths, sid, ja_bert, en_bert)  # , emo
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return audio


