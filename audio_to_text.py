import os
import re
import speech_recognition as sr
from pydub import AudioSegment
import pandas as pd
import jieba
from jieba import posseg
from snownlp import SnowNLP
from collections import defaultdict
import numpy as np
from tqdm import tqdm


# 加载中文停用词
stopwords_file = "stopwords.txt"
stopwords = []
if os.path.exists(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()


def read_audio_file(audio_path):
    """读取音频文件"""
    return AudioSegment.from_file(audio_path)


def split_audio_into_segments(audio, segment_length=10 * 1000):
    """将音频分割成指定长度的片段"""
    segments = []
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i + segment_length]
        segments.append(segment)
    return segments


def transcribe_segment(segment):
    """对单个音频片段进行转录"""
    r = sr.Recognizer()
    with sr.AudioFile(segment) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data, language='zh-CN')
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"请求错误：{e}")
            return ""


def transcribe_audio(segments):
    """对所有音频片段进行转录并合并文本"""
    full_text = ""
    for segment in tqdm(segments, desc="Transcribing audio"):
        segment_path = "temp_segment.wav"
        segment.export(segment_path, format="wav")
        text = transcribe_segment(segment_path)
        full_text += text
        os.remove(segment_path)
    return full_text


def clean_text(text):
    """清理文本，去除无关信息、口语化表达和噪声数据"""
    # 去除一些常见的口语化表达
    text = re.sub(r'\嗯+|\呃+|那个|就是|然后', '', text)
    # 去除一些无关的语气词
    text = re.sub(r'啊|呀|吧|呢|啦', '', text)
    # 去除连续的空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_text_by_question(text):
    """按问题将文本分开"""
    pattern = re.compile(r'第(\d+)个问题')
    matches = pattern.finditer(text)
    split_texts = []
    start = 0
    for match in matches:
        split_texts.append(text[start:match.start()].strip())
        start = match.start()
    split_texts.append(text[start:].strip())
    # 去除空的部分
    split_texts = [s for s in split_texts if s]
    return split_texts


def split_text_into_paragraphs(split_text):
    """将按问题分开的文本再按句子分段"""
    all_paragraphs = []
    for sub_text in split_text:
        sentences = []
        current_sentence = ""
        for char in sub_text:
            if char in ['。', '！', '？']:
                current_sentence += char
                sentences.append(current_sentence)
                current_sentence = ""
            else:
                current_sentence += char
        if current_sentence:
            sentences.append(current_sentence)

        paragraphs = []
        current_paragraph = ""
        for sentence in sentences:
            if len(current_paragraph) + len(sentence) < 200:
                current_paragraph += sentence
            else:
                paragraphs.append(current_paragraph.strip())
                current_paragraph = sentence
        if current_paragraph:
            paragraphs.append(current_paragraph.strip())
        all_paragraphs.extend(paragraphs)
    return all_paragraphs


def preprocess_text(text):
    """中文文本预处理，包括分词、去停用词"""
    words = jieba.lcut(text)
    filtered_words = [word for word in words if word not in stopwords]
    return " ".join(filtered_words)


def sentiment_analysis(text):
    """进行情感识别"""
    s = SnowNLP(text)
    return s.sentiments


def process_audio_file(audio_path, participant_id):
    """处理音频文件的主函数"""
    audio = read_audio_file(audio_path)
    segments = split_audio_into_segments(audio)
    full_text = transcribe_audio(segments)
    cleaned_text = clean_text(full_text)
    split_by_question = split_text_by_question(cleaned_text)

    all_data = []
    for question_id, sub_text in enumerate(split_by_question, 1):
        sentences = []
        current_sentence = ""
        for char in sub_text:
            if char in ['。', '！', '？']:
                current_sentence += char
                sentences.append(current_sentence)
                current_sentence = ""
            else:
                current_sentence += char
        if current_sentence:
            sentences.append(current_sentence)

        for sentence in sentences:
            preprocessed_sentence = preprocess_text(sentence)
            sentiment = sentiment_analysis(preprocessed_sentence)
            all_data.append([participant_id, question_id, sentence, preprocessed_sentence, sentiment])

    return all_data


def generate_dataframe(data):
    """将处理后的数据生成DataFrame"""
    columns = ['ParticipantID', 'QuestionID', 'SentenceText', 'PreprocessedSentence', 'Sentiment']
    df = pd.DataFrame(data, columns=columns)
    return df


def save_to_csv(df, output_path='processed_data.csv'):
    """将DataFrame保存为CSV文件"""
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    audio_path = "interview_audio.wav"  # 替换为你的访谈录音文件路径
    participant_id = 1  # 替换为实际的参与者ID
    processed_data = process_audio_file(audio_path, participant_id)
    df = generate_dataframe(processed_data)
    save_to_csv(df)
    print("数据已成功保存为 processed_data.csv")
