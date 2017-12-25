#将文件切分成N个小文件
def segFile(srcPath,resPath,N):
    f = open(srcPath,'r')
    lines = f.readlines()
    l = len(lines)
    part = l / N
    for i in range(N):
        f_res = open(resPath + str(N) + '.txt', 'w', encoding='utf8')
        if i == N - 1:
            f_res.writelines(lines[part * i:])
        else:
            f_res.writelines(lines[part * i:part * (i + 1)])

if __name__ == "__main__":
    srcPath = 'E:\Data\EmojiPrediction\\emoji_sample_withBlankbeforePunc.txt'
    resPath = 'E:\Data\EmojiPrediction\\emoji_sample_withBlankbeforePunc'
    N = 10
    segFile(srcPath,resPath,N)