import re
import string
import nltk
from nltk.corpus import stopwords
import ptb.conf as conf

#字母数字，标点符号，emojis除外都过滤掉，包括停用词
def filterSpecCha(srcPath, resPath):
    f_src = open(srcPath,'r',encoding='utf8')
    srcLines = f_src.readlines()
    res = []
    regexPunc = string.punctuation + '。\，'
    count = 0
    nltk.download('stopwords')
    english_stopwords = stopwords.words('english')
    print(len(english_stopwords))

    for line in srcLines:
        resLine = ""
        print(count)
        count += 1
        items = line.split()
        for item in items:
            if item in conf.emojiList:#emojis
                resLine = resLine + item + ' '
            elif item in regexPunc:#标点符号
                resLine = resLine + item + ' '
            elif re.match(r'[a-zA-Z0-9]+', item) and item not in english_stopwords:#至少有一个数字或者英文字符,且单词不在停用词表中
                # elif re.match(r'[a-zA-Z0-9]+', item) and item not in english_stopwords:  # 至少有一个数字或者英文字符,且单词不在停用词表中
                resLine = resLine + item + " "
            else:
                resLine = resLine + "<unk> "
        res.append(resLine.strip() + '\n')
    f_res = open(resPath,'w',encoding='utf8')
    f_res.writelines(res)

if __name__ == "__main__":


    srcPath = 'E:\Data\EmojiPrediction\\emoji_sample_withBlankbeforePunc_head.txt'
    resPath = 'E:\Data\EmojiPrediction\\emoji_sample_withBlankbeforePunc_head_filter.txt'

    # srcPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge.txt'
    # resPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter.txt'
    filterSpecCha(srcPath, resPath)
    print('Finished!')