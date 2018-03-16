import ptb.conf as conf
import os
#将文件切分成N个小文件
def segFile(srcPath,resPath,N):
    f = open(srcPath,'r',encoding='utf8')
    lines = f.readlines()
    l = len(lines)
    part = int(l / N)
    for i in range(N):
        f_res = open(resPath + '/' + str(i) + '.txt', 'w', encoding='utf8')
        # f_res = open(resPath + '\\' + str(i) + '.txt', 'w', encoding='utf8')
        if i == N - 1:
            f_res.writelines(lines[part * i:])
        else:
            f_res.writelines(lines[part * i:part * (i + 1)])

if __name__ == "__main__":
    # srcPath = conf.src_path + '\\emoji_sample_withBlankbeforePunc.txt'
    # resPath = conf.src_path + '\\emoji_sample_withBlankbeforePunc'
    # srcPath = conf.src_path + '/emoji_sample_withBlankbeforePunc.txt'
    # resPath = conf.src_path + '/emoji_sample_withBlankbeforePunc'
    srcPath = conf.src_path + '/emoji_sample.txt'
    resPath = conf.src_path + '/emoji_sample'
    if not os.path.exists(resPath):
        os.mkdir(resPath)
        print('Successfully created directory', resPath)
    N = 10
    segFile(srcPath,resPath,N)
    print('Finished!')