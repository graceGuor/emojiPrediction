import nltk
from nltk.corpus import stopwords
import ptb.conf as conf
import sys

#去掉文本中的停用词
def removeStopwords(srcPath,resPath):
    f_src = open(srcPath, 'r', encoding='utf8')
    srcLines = f_src.readlines()
    res = []
    count = 0
    # english_stopwords = stopwords.words('english')
    # print(len(english_stopwords))

    for line in srcLines:
        resLine = ""
        # print(count)
        count += 1
        items = line.split()
        for item in items:
            if item not in conf.english_stopwords:  # 单词不在停用词表中
                resLine = resLine + item + " "
            else:
                resLine = resLine + "<unk> "
        res.append(resLine.strip() + '\n')
    f_res = open(resPath, 'w', encoding='utf8')
    f_res.writelines(res)

def removeStopwords_main():
    srcPath = 'E:\Data\EmojiPrediction\\testGr.txt'
    resPath = 'E:\Data\EmojiPrediction\\testGrRes.txt'
    srcPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter_lower.txt'
    resPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter_lower_stopwords.txt'
    removeStopwords(srcPath, resPath)

if __name__ == "__main__":
    removeStopwords_main()
    print(sys.argv[0] + 'Finished!')




