from gensim.models.fasttext import FastText as FT_gensim
from gensim.models.word2vec import LineSentence
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

pretrained_model_path = "../models/transformed.model"
processed_data_path = "../datasets/weibo/dataset_cut.txt"
post_trained_model_path = "../models/trained_weibo.model"

model = FT_gensim.load(pretrained_model_path)

model.build_vocab(LineSentence(open(processed_data_path, 'r', encoding='utf8')),
                  update=True)
model.train(
    LineSentence(open(processed_data_path, 'r', encoding='utf8')),
    epochs=1,
    total_examples=model.corpus_count,
    total_words=model.corpus_total_words,
    )

model.save(post_trained_model_path)
