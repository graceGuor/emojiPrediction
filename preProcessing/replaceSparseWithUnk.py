import tensorflow as tf
import collections
import ptb.conf as conf

#将数据集中出现频率除TopK之外的词都用<unk>代替
def replaceSparseWithUnk(srcPath, resPath):
    f_src = tf.gfile.GFile(srcPath,'r')
    data = f_src.read().replace("\n","<eos> ").split()
    counter = collections.Counter(data)
    print("all words size:" + str(len(counter)))
    count_pairs = sorted(counter.items(),reverse=True,key=lambda x:(-x[1],x[0]))
    words,_ = list(zip(*count_pairs[0:conf.vocab_size]))#算进去了<eos>，<unk>
    print('word_dict.size:' + str(len(words)))

    res = []
    f_sour = open(srcPath,'r',encoding='utf8')
    lines = f_sour.readlines()
    for line in lines:
        resLine = ""
        items = line.split()
        for item in items:
            if item in words:
                resLine = resLine + item + " "
            else:
                resLine = resLine + "<unk> "
        # print(resLine)
        res.append(resLine.strip() + '\n')
    f_res = open(resPath,'w',encoding='utf8')
    f_res.writelines(res)

if __name__ == "__main__":
    srcPath = 'E:\Data\EmojiPrediction\emoji_sample_head.txt'
    resPath = 'E:\Data\EmojiPrediction\emoji_sample_head_unk.txt'
    replaceSparseWithUnk(srcPath,resPath)
    print('Finished!')