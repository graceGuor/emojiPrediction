#给文件中的每一行都添加引号，给行与行之间添加逗号
def addQuotationMark(srcPath, resPath):
    f_src = open(srcPath, 'r', encoding='utf8')
    srcLines = f_src.readlines()
    res = ''
    for line in srcLines:
        line = '"' + line.strip() + '",'
        res = res + line
    f_res = open(resPath, 'w', encoding='utf8')
    f_res.writelines(res)

if __name__ == "__main__":
    srcPath = 'E:\Data\EmojiPrediction\english.txt'
    resPath = 'E:\Data\EmojiPrediction\english_stopwords.txt'
    addQuotationMark(srcPath, resPath)
    print('Finished!')