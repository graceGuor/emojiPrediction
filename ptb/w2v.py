from gensim.models import word2vec
import gensim
import os
import ptb.conf as conf
import numpy as np
import pdb

train_path = os.path.join(conf.data_path, "train.txt")
f = open(train_path, 'r', encoding='utf-8')
sentences = f.readlines()
sentences = [sentence.split() for sentence in sentences]
# sentences = [['first', 'sentence'], ['second', 'sentence']]
model = word2vec.Word2Vec(sentences,
                          min_count=conf.min_count,
                          size=conf.embedding_size)
embs = []
# model.save(conf.emb_model_savePath)
# new_model = gensim.models.Word2Vec.load(conf.emb_model_savePath)
# print('new model')
# print(new_model['first'])
for word in model.wv.vocab:
    emb = word + " " + str(model[word]).replace('\n', '')[1:]
    embs.append(emb[0:len(emb)-1])
f_res = open(conf.emb_path, 'w', encoding='utf-8')
s = '\n'.join(embs)
f_res.writelines(s)
f_res.close()
# np.savetxt(conf.save_path + '/emb.txt', model.wv.syn0)
print('Finished!')