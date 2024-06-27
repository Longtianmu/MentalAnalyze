from utils.common_words import risk_connection_words
from utils.common_words import common_connection_word
import pkuseg
import re


def check_regex(word):
    res = False
    res = res and ('' == re.sub(r'[^\\u4e00-\\u9fff]+', '', word))
    return res


def analyze_single_sentence(input_sentence, model, cut_model_path, stop_words, debug_mode=False):
    used_tokens = []
    input_tokens = pkuseg.pkuseg(cut_model_path).cut(input_sentence)

    for i in input_tokens:
        if check_regex(i):
            continue
        if i in stop_words:
            continue
        i = re.sub("[^\u4e00-\u9fa5^]", "", i)
        if i == "":
            continue
        used_tokens.append(i)

    if len(used_tokens) == 0:
        print("Sentence may not have actual meaning or else.")
        return 0
    else:
        total_negative_value = 0.0
        total_positive_value = 0.0
        tokens_count = 0
        # Calc Total Similarity
        for token in used_tokens:
            negative_similarity = []
            positive_similarity = []

            for negative in risk_connection_words:
                similarity = 0.0

                for similar_words in negative[1]:
                    similarity += (model.wv.similarity(token, similar_words))

                current_tuple = (negative[0], similarity)
                negative_similarity.append(current_tuple)

            for positive in common_connection_word:
                similarity = 0.0

                for similar_words in positive[1]:
                    similarity += (model.wv.similarity(token, similar_words))

                current_tuple = (positive[0], similarity)
                positive_similarity.append(current_tuple)

            negative_value = 0.0
            positive_value = 0.0
            for i in negative_similarity:
                negative_value += i[1]
            for i in positive_similarity:
                positive_value += i[1]

            if negative_value == 0 or positive_value == 0:
                print(f"No NGRAMS found for token {token}, skipped")
                continue
            tokens_count += 1
            total_negative_value += negative_value
            total_positive_value += positive_value
            if debug_mode:
                print(token, negative_value, positive_value)
                print("")

            if tokens_count == 0:
                print("Sentence may not have actual meaning or else.")
                return 0

        total_negative_value = total_negative_value / tokens_count
        total_positive_value = total_positive_value / tokens_count

        return total_negative_value, total_positive_value
