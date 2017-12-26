import ptb.conf as conf
import os

#将N个小文件合并成一个文件
def segFile(srcPath,resPath,N):
    res = []
    for i in range(N):
        # f = open(srcPath + '\\' + str(i) + '.txt', 'r', encoding='utf8')
        f = open(srcPath + '/' + str(i) + '.txt', 'r', encoding='utf8')
        lines = f.readlines()
        res += lines
        # res.append(lines)
    f_res = open(resPath, 'w', encoding='utf8')
    f_res.writelines(res)

if __name__ == "__main__":
    # srcPath = conf.src_path + '\\testGr'
    # resPath= conf.src_path + '\\testGr_merge.txt'

    srcPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo'
    resPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge.txt'

    N = 10
    segFile(srcPath,resPath,N)
    print('Finished!')