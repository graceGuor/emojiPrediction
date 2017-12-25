import tensorflow as tf
import collections
import ptb.conf as conf

#将数据集中出现频率除TopK之外的词都用<unk>代替
def replaceSparseWithUnk(srcPath, resPath):
    f_src = tf.gfile.GFile(srcPath,'r')
    data = f_src.read().replace("\n","<eos>").split()
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(),reverse=True,key=lambda x:(-x[1],x[0]))
    words,_ = list(zip(*count_pairs[0:conf.vocab_size - 1]))

    res = []
    count = 1

    for line in srcLines:
        resLine = ""

        # print(resLine)
        res.append(resLine.strip() + '\n')
    f_res = open(resPath,'w',encoding='utf8')
    f_res.writelines(res)

if __name__ == "__main__":
    srcPath = 'E:\Data\EmojiPrediction\emoji_Unicode_2623.txt'
    resPath = 'E:\Data\EmojiPrediction\emoji_Unicode.txt'
    replaceSparseWithUnk(srcPath,resPath)