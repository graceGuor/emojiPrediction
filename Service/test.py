import numpy as np
import tensorflow as tf
import collections
import pdb
from nltk.corpus import stopwords
import ptb.conf as conf

english_stopwords = stopwords.words('english')
print(len(english_stopwords))
src = "💋💕 👶🏻   😂 。 ， test 发 i'm used ! * Please  tɦaռҡs"
l = len(src)
for i in range(l):
    if src[i] in conf.emojiList:
        print(src[i])

# print(isEmoji(r'\U0001F600'))