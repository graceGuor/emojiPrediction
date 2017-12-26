import re
import string
import ptb.conf as conf

#字母数字，标点符号，emojis除外都过滤掉
def filterSpecCha(srcPath, resPath):
    f_src = open(srcPath,'r',encoding='utf8')
    srcLines = f_src.readlines()
    res = []
    regexPunc = string.punctuation + '。\，'

    for line in srcLines:
        resLine = ""
        print(line)
        items = line.split()
        for item in items:
            if item in conf.emojiList:#emojis
                resLine = resLine + item + ' '
            elif item in regexPunc:#标点符号
                resLine = resLine + item + ' '
            elif re.match(r'[a-zA-Z0-9]+', item):#至少有一个数字或者英文字符
                resLine = resLine + item + " "
            else:
                resLine = resLine + "<unk> "
        print(resLine)
        res.append(resLine.strip() + '\n')
    f_res = open(resPath,'w',encoding='utf8')
    f_res.writelines(res)

if __name__ == "__main__":
    srcPath = 'E:\Data\EmojiPrediction\\testGr.txt'
    resPath = 'E:\Data\EmojiPrediction\\testGrRes.txt'

    srcPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge.txt'
    resPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter.txt'
    filterSpecCha(srcPath, resPath)
    print('Finished!')