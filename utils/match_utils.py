import re


def verify_text_format(input_text):
    # 验证说话人的正则表达式
    pattern_speaker = r"(\[\S+?\])((?:\s*<\S+?>[^<\[\]]+?)+)"

    # 使用re.DOTALL标志使.匹配包括换行符在内的所有字符
    matches = re.findall(pattern_speaker, input_text, re.DOTALL)

    # 对每个匹配到的说话人内容进行进一步验证
    for speaker, dialogue in matches:
        language_text_matches = match_language(speaker, dialogue)
        if not language_text_matches:
            return False, "Error: Invalid format detected in dialogue content. Please check your input."
    # 如果输入的文本中没有找到任何匹配项
    if not matches:
        return False, "Error: No valid speaker format detected. Please check your input."
    return True, "Input is valid."


def match_speaker_language(text: str) -> list:
    # 提取说话者和对应文本
    speaker_table_text = match_speaker(text)
    result = []
    # result包含多个元组的列表，每个元组包含一个说话者标签（如[Alice]）, 语言和一个对话字符串[(语言, 文本), (语言, 文本), ..., 说话者]
    for speaker, dialogue in speaker_table_text:
        one_speaker_list = match_language(speaker, dialogue)
        result.append(one_speaker_list)
    return result


def match_speaker(text):
    # 提取说话者和对应文本
    speaker_pattern = r"(\[\S+?\])(.+?)(?=\[\S+?\]|$)"
    matches = re.findall(speaker_pattern, text, re.DOTALL)
    return matches


def match_language(speaker, dialogue):
    # 使用正则表达式匹配<语言>标签和其后的文本
    pattern_language_text = r"<(\S+?)>([^<]+)"
    matches = re.findall(pattern_language_text, dialogue, re.DOTALL)
    speaker = speaker[1:-1]
    # 清理文本：去除两边的空白字符
    matches_truple = [(lang, text.strip()) for lang, text in matches]
    matches_truple.append(speaker)  # 清理后的 语言标签、文本和说话者标签, [(语言, 文本), (语言, 文本), ..., 说话者]
    return matches_truple


def split_sentences(para):
    # 通过正则表达式为标点符号后面添加换行符，然后通过换行符来进行分割
    para = re.sub("([。！;？\?])([^”’])", r"\1\n\2", para)  # 单字符断句符
    para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)  # 英文省略号
    para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)  # 中文省略号
    para = re.sub("([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")


def split_paragraphs(text):
    # 将文本按换行符(段落)分割成段落并删除空字符串
    split_para = [sentence.strip() for sentence in re.split(r"\n+", text) if sentence.strip()]
    return split_para


if __name__ == "__main__":
    text1 = """
    [说话人1]
    [说话人2]<zh>你好吗？<jp>元気ですか？<jp>こんにちは，世界。<zh>你好吗？
    [说话人3]<zh>谢谢。<jp>どういたしまして。
    """
    text_result = match_speaker_language(text1)
    # 测试函数
    text2 = """
    [说话人1]<zh>你好，こんにちは！<jp>こんにちは，世界。
    [说话人2]<zh>你好吗？
    """
    test_text_result = match_speaker_language(text2)
    res = verify_text_format(text2)
    print("text_result:", text_result, "\ntest_text_result:", test_text_result, "\n", res)
