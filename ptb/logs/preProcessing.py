import re
import string
import pdb

def testFilterSpecCha():
    srcPath = 'E:\Data\EmojiPrediction\\testGr.txt'
    resPath = 'E:\Data\EmojiPrediction\\testGrRes.txt'
    emojiFile = 'E:\Data\EmojiPrediction\emoji_Unicode.txt'
    filterSpecCha(srcPath, resPath, emojiFile)

#字母数字，标点符号，emojis除外都过滤掉
def filterSpecCha(srcPath, resPath, emojiFile):
    f_src = open(srcPath,'r',encoding='utf8')
    srcLines = f_src.readlines()
    res = []
    regexPunc = string.punctuation + '。\，'
    emojis = open(emojiFile, 'r', encoding='utf8').readlines()

    # emojis = [emoji.strip() for emoji in emojis]
    for emoji in emojis:
        emoji = emoji.strip()

        # emoji = chr(int(emoji[2:], 16))

    for line in srcLines:

        resLine = ""
        print(line)
        items = line.split()

        for item in items:
            # if item in ['\U0001F476\U0001F3FB']:#true
            #     print("bb")
            # emojis_test = ['\U0001F476\U0001F3FB']
            # print(len(emojis_test[0]))
            # if emojis_test == ['\U0001F476\U0001F3FB']:#true
            #     print("ff")
            # if item == '\U0001F476\U0001F3FB':#true
            #     print("cc")

            # s = r'\U0001F476\U0001F3FB' # str
            # us = '\U0001F476\U0001F3FB' # emoji
            # s1 = r'\U0001F476'
            # s2 = r'\U0001F3FB'
            # s2us = chr(int(s1[2:], 16)) + chr(int(s2[2:], 16))  # 是emoji
            # us2s = ord(us)  # 是数值

            # s2us = chr(int(s[2:10], 16) + int(s[12:], 16))  # 是emoji
            # # s2us = chr(int(s[2:], 16))  # 是emoji
            # us2s = ord(us)  # 是数值
            # print(s2us, us2s,us)


            if item in emojis:#emojis
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
    testFilterSpecCha()
    print("Finished！")