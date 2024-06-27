import pkuseg
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

seg = pkuseg.pkuseg(model_name="../models/pkuseg/default_v2")
sentence = input("Please input sentence:")
sentence = seg.cut(sentence)
print("，".join(sentence))
