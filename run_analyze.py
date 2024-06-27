from utils.analyze_sentence import analyze_single_sentence
from gensim.models.fasttext import FastText as FT_gensim
import csv
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

vec_model_path = "models/transformed.model"
cut_model_path = "models/pkuseg/default_v2"
stop_word_path = "datasets/stop_words_detect.txt"

model = FT_gensim.load(vec_model_path)
stop_words = []
with open(stop_word_path, "r", encoding="utf-8") as input_file:
    for line in input_file:
        res = line.strip()
        if line == "\n":
            continue
        stop_words.append(res)

validation_set_path = "datasets/validation/set1.csv"
stop_word_path = "datasets/stop_words_detect.txt"

stop_words = []
with open(stop_word_path, "r", encoding="utf-8") as input_file:
    for line in input_file:
        res = line.strip()
        if line == "\n":
            continue
        stop_words.append(res)

input_data = []
with open(validation_set_path, encoding='utf-8') as input_file:
    for row in csv.reader(input_file):
        if row == ["status", "text"]:
            continue
        input_data.append(row)

risk_true = 0
common_true = 0
risk_false = 0
common_false = 0
count = 0
total_count = len(input_data)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

for single_data in input_data:
    count += 1
    result = int(single_data[0])
    single_sentence = single_data[1]
    res = analyze_single_sentence(single_sentence, model, cut_model_path, stop_words, debug_mode=True)
    risk_points = res[0]
    common_points = res[1]
    print(f"Complete Sentence {count}/{total_count}, Risk {risk_points}, Common {common_points}")

    if risk_points > common_points:
        res = 1
        print(f"Predicted {res}, Fact {result}")
        if result == res:
            risk_true += 1
        else:
            risk_false += 1
    else:
        res = 0
        print(f"Predicted {res}, Fact {result}")
        if result == res:
            common_true += 1
        else:
            common_false += 1
print("Risk True:", risk_true)
print("Risk False:", risk_false)
print("Common True:", common_true)
print("Common False:", common_false)

precision_rate = risk_true / (risk_true + risk_false)
print("Precision Rate:", precision_rate)
recall_rate = risk_true / (risk_true + common_false)
print("Recall Rate:", recall_rate)
f1_score = 2.0 * precision_rate * recall_rate / (precision_rate + recall_rate)
print("F1_Score:", f1_score)
accuracy_rate = (risk_true + common_true) / total_count
print("Accuracy Rate:", accuracy_rate)
