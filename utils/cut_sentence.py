from analyze_sentence import check_regex
import pkuseg
import logging
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

stop_word_path = "../datasets/stop_words_cn.txt"
input_file_path = "../datasets/weibo/dataset.txt"
output_file_path = "../datasets/weibo/dataset_cut.txt"
seg = pkuseg.pkuseg(model_name="../models/pkuseg/default_v2")
stop_words = []

with open(stop_word_path, "r", encoding="utf-8") as input_file:
    for line in input_file:
        res = line.strip()
        if line == "\n":
            continue
        stop_words.append(res)
lines = []
with open(input_file_path, "r", encoding="utf-8") as input_file:
    for line in input_file:
        if line == "\n":
            continue
        res = line.strip()
        res = seg.cut(res)

        lines.append(res)

with open(output_file_path, "w", encoding="utf-8") as output_file:
    for line in lines:
        sentence = []
        for i in line:
            if check_regex(i):
                continue
            if i in stop_words:
                continue
            i = re.sub("[^\u4e00-\u9fa5^]", "", i)
            if i == "":
                continue
            sentence.append(i)
        res = " ".join(sentence)
        if res == "" or res == "\n":
            continue
        output_file.write(res + "\n")
