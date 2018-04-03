import ptb.conf as conf
import ptb.reader as reader
import sys
import tensorflow as tf
import os
import re

#得到每个词与emoji共现的特征
def getEmojiCoOccur(srcPath, resPath1, resPath3, resPath5):
    word_to_id = reader._build_vocab(srcPath)
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    # print(word_to_id)
    # print(id_to_word)
    f = tf.gfile.GFile(srcPath, "r")
    data = f.read().replace("\n", " <eos> ").split()
    data = [word_to_id[word] for word in data if word in word_to_id]
    # print(data)
    lenOfWordDict = len(word_to_id)
    idOfFirstWord = 327#43#
    # count = [0] * idOfFirstWord
    res_fea1 = []
    res_fea3 = []
    res_fea5 = []

    for i in range(lenOfWordDict):
        # res_fea1.append(count)#所有结果都一样
        res_fea1.append([0] * idOfFirstWord)
        res_fea3.append([0] * idOfFirstWord)
        res_fea5.append([0] * idOfFirstWord)
    # res_fea[0][1] = res_fea[0][1] + 1
    # print(res_fea)
    res_matric1 = dict(zip(word_to_id.keys(), res_fea1))
    res_matric3 = dict(zip(word_to_id.keys(), res_fea3))
    res_matric5 = dict(zip(word_to_id.keys(), res_fea5))
    # print(res_matric1)

    for i in range(len(data)):
        if data[i] < idOfFirstWord:
            # print(data[i])
            # print(id_to_word[data[i]])
            # print(res_matric1[id_to_word[data[i]]])
            # print(res_matric1[id_to_word[data[i]]][data[i]])
            for j in range(-5, 1):
                if j in range(-1, 2) and i+j in range(0, len(data)):
                    res_matric1[id_to_word[data[i + j]]][data[i]] += 1
                    res_matric3[id_to_word[data[i + j]]][data[i]] += 1
                    res_matric5[id_to_word[data[i + j]]][data[i]] += 1
                elif j in range(-3, 4) and i+j in range(0, len(data)):
                    res_matric3[id_to_word[data[i + j]]][data[i]] += 1
                    res_matric5[id_to_word[data[i + j]]][data[i]] += 1
                else:
                    # print(data[i + j])
                    # print(id_to_word[data[i + j]])
                    # print(res_matric5[id_to_word[data[i + j]]])
                    res_matric5[id_to_word[data[i + j]]][data[i]] += 1
                    # print(res_matric3[id_to_word[data[i + j]]])
                    # print(res_matric5[id_to_word[data[i + j]]])

    # print(res_matric1)
    print('word size:' + str(len(res_matric1)))
    p2 = re.compile("[\[{}\']")
    res1 = p2.sub("", str(res_matric1)).replace('], ', '\n').replace(":", " ").replace(",", " ").replace("]", "")
    res3 = p2.sub("", str(res_matric3)).replace('], ', '\n').replace(":", " ").replace(",", " ").replace("]", "")
    res5 = p2.sub("", str(res_matric5)).replace('], ', '\n').replace(":", " ").replace(",", " ").replace("]", "")
    f_res1 = open(resPath1, 'w', encoding='utf8')
    f_res1.writelines(res1)
    f_res3 = open(resPath3, 'w', encoding='utf8')
    f_res3.writelines(res3)
    f_res5 = open(resPath5, 'w', encoding='utf8')
    f_res5.writelines(res5)
    return  res_matric1, res_matric3, res_matric5

def getEmojiCoOccur_main():
    srcPath = os.path.join(conf.data_path, "train.txt")
    # data_path = conf.src_path + "\Fold_head\/all"
    # srcPath = os.path.join(data_path, "train.txt")
    resPath1 = os.path.join(conf.src_path, "emoji_coOccur1_fea.txt")
    resPath3 = os.path.join(conf.src_path, "emoji_coOccur3_fea.txt")
    resPath5 = os.path.join(conf.src_path, "emoji_coOccur5_fea.txt")
    res_matric1, res_matric3, res_matric5 = getEmojiCoOccur(srcPath, resPath1, resPath3, resPath5)

if __name__ == "__main__":
    # print(RI.getAverageLenOfSentences(conf.alldata_path))
    getEmojiCoOccur_main()
    print(sys.argv[0] + ' Finished!')