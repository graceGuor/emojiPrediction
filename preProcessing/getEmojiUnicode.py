import pdb

#源文件中有些行有多个emoji的Unicode，如“\U0001F443\U0001F3FD”，把它们拆分成一行只有一个Unicode
#不能这样，有些emoji的Unicode编码是多个的组合
def getEmojiUnicode(srcPath, resPath):
    f_src = open(srcPath,'r',encoding='utf8')
    firstLine = f_src.readline()
    l = len(firstLine)
    srcLines = f_src.readlines()
    res = []
    for line in srcLines:
        if len(line) == l:
            res.append(line)
        else:
            # pdb.set_trace()
            items = line.split("\\U")
            for item in items:
                if len(item) != 0:
                    item = "\\U" + item.strip()
                    res.append(item)

    f_res = open(resPath, 'w', encoding='utf8')
    f_res.writelines(res)

if __name__ == "__main__":
    # print('\U0001F476\U0001F3FB')
    srcPath = 'E:\Data\EmojiPrediction\emoji_Unicode_2623.txt'
    resPath = 'E:\Data\EmojiPrediction\emoji_Unicode.txt'
    getEmojiUnicode(srcPath, resPath)
    print('Finished!')
