from utils.analyze_sentence import analyze_single_sentence
from gensim.models.fasttext import FastText as FT_gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
vec_model_path = "models/transformed.model"
cut_model_path = "models/pkuseg/default_v2"
model = FT_gensim.load(vec_model_path)

stop_word_path = "datasets/stop_words_detect.txt"
stop_words = []
with open(stop_word_path, "r", encoding="utf-8") as input_file:
    for line in input_file:
        res = line.strip()
        if line == "\n":
            continue
        stop_words.append(res)


inputs = input("Please input sentence for check: ")
print(analyze_single_sentence(inputs, model, cut_model_path, stop_words, True))
