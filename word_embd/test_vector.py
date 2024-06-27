from gensim.models.fasttext import FastText as FT_gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

pre_trained_model_path = "../models/transformed.model"
post_trained_model1_path = "../models/trained_mixed.model"
post_trained_model2_path = "../models/trained_weibo.model"
model_path = pre_trained_model_path
test_word = "抑郁"
test_word2 = "绝望"

input_word = "烧炭"

model = FT_gensim.load(model_path)
loaded_model = model.wv
print(loaded_model.most_similar(test_word))
