import re
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def format_text(text):
    pattern_all = re.compile("[^\u4e00-\u9fa5^]")
    cop = re.sub(pattern_all, "", text)
    return cop


input_file_path = "../datasets/mixed/mixed_translated.txt"
output_file_path = "../datasets/mixed/dataset_processed.txt"

lines = []

with open(input_file_path, "r", encoding="utf-8") as input_file:
    for line in input_file:
        if line == "\n":
            continue
        res = line.strip()
        res = format_text(res)

        lines.append(res)

with open(output_file_path, "w", encoding="utf-8") as output_file:
    for line in lines:
        if line == "\n":
            continue
        if line == "":
            continue
        output_file.write(format_text(line) + "\n")
