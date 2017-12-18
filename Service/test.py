import numpy as np
import pdb

# words = {"sam","tom"}
words = [1,2]
# b = [str(i) for i in words]
words = np.reshape(words,[2,1])
fea = [[1,2],[3,4]]
fe = np.concatenate([words,fea], axis=1)
res = []
for f in fe:
    res.append([str(i) for i in f])
# pdb.set_trace()
# fe = map(str,fea)
em = list(zip(list(words) ,fea))
emb = dict(zip(words,fea))
word_id = {"sam":1,"tomm":2}
for (k,v) in emb.items():
    len = len(v)
    break
for (k,v) in word_id.items():
    if k in emb:
        print(emb[k])
    else:
        print(np.random.rand(len))