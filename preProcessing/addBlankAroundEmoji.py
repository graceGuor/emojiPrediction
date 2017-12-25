import re
import string
import ptb.conf as conf

def testAddBlankAroundEmoji():
    srcPath = 'E:\Data\EmojiPrediction\\emoji_sample_withBlankbeforePunc.txt'
    resPath = 'E:\Data\EmojiPrediction\\emoji_sample_withBlankbeforePunc_blankEmo.txt'
    addBlankAroundEmoji(srcPath, resPath)

#字母数字，标点符号，emojis除外都过滤掉
def addBlankAroundEmoji(srcPath, resPath):
    f_src = open(srcPath,'r',encoding='utf8')
    srcLines = f_src.readlines()
    res = []
    count = 1

    for line in srcLines:
        resLine = ""
        print(count)
        count = count + 1
        l = len(line)
        i = 0
        while(i < l):
            if i < l - 7 and line[i] + line[i + 1] + line[i + 2] + line[i + 3] + line[i + 4] + line[i + 5] + line[i + 6] + line[i + 7] in conf.emojiList:
                resLine = resLine + " " + line[i] + line[i + 1] + line[i + 2] + line[i + 3] + line[i + 4] + line[i + 5] + line[i + 6] + line[i + 7] + " "
                i = i + 8
            elif i < l - 6 and line[i] + line[i + 1] + line[i + 2] + line[i + 3] + line[i + 4] + line[i + 5] + line[i + 6] in conf.emojiList:
                resLine = resLine + " " + line[i] + line[i + 1] + line[i + 2] + line[i + 3] + line[i + 4] + line[i + 5] + line[i + 6] + " "
                i = i + 7
            elif i < l - 5 and line[i] + line[i + 1] + line[i + 2] + line[i + 3] + line[i + 4] + line[i + 5] in conf.emojiList:
                resLine = resLine + " " + line[i] + line[i + 1] + line[i + 2] + line[i + 3] + line[i + 4] + line[i + 5] + " "
                i = i + 6
            elif i < l - 4 and line[i] + line[i + 1] + line[i + 2] + line[i + 3] + line[i + 4] in conf.emojiList:
                resLine = resLine + " " + line[i] + line[i + 1] + line[i + 2] + line[i + 3] + line[i + 4] + " "
                i = i + 5
            elif i < l - 3 and line[i] + line[i + 1] + line[i + 2] + line[i + 3] in conf.emojiList:
                resLine = resLine + " " + line[i] + line[i + 1] + line[i + 2] + line[i + 3] + " "
                i = i + 4
            elif i < l - 2 and line[i] + line[i + 1] + line[i + 2] in conf.emojiList:
                resLine = resLine + " " + line[i] + line[i + 1] + line[i + 2] + " "
                i = i + 3
            elif i < l - 1 and line[i] + line[i + 1] in conf.emojiList:
                resLine = resLine + " " + line[i] + line[i + 1] + " "
                i = i + 2
            elif line[i] in conf.emojiList:
                resLine = resLine + " " + line[i] + " "
                i = i + 1
            else:
                resLine = resLine + line[i]
                i = i + 1
        # print(resLine)
        res.append(resLine.strip() + '\n')
    f_res = open(resPath,'w',encoding='utf8')
    f_res.writelines(res)

if __name__ == "__main__":
    testAddBlankAroundEmoji()