import ptb.conf as conf
import os
import sys

#对数据集进行划分
def dataDivision811(srcPath,resPath, fold):
    f_sour = open(srcPath,'r',encoding='utf8')
    lines = f_sour.readlines()
    print('all data size:' + str(len(lines)))
    sizeOfFold = int(len(lines)/fold)
    print('size of each fold:' + str(sizeOfFold))

    f = 0

    # if not os.path.exists(resPath + '\\' + str(f)):
    #     os.mkdir(resPath + '\\' + str(f))
    #     print('Successfully created directory', resPath + '\\' + str(f))
    # f_res = open(resPath + '\\' + str(f) + '\\test.txt', 'w', encoding='utf8')
    # f_res.writelines(lines[:sizeOfFold])
    # f_res = open(resPath + '\\' + str(f) + '\\validation.txt', 'w', encoding='utf8')
    # f_res.writelines(lines[sizeOfFold:sizeOfFold * 2])
    # f_res = open(resPath + '\\' + str(f) + '\\train.txt', 'w', encoding='utf8')
    # f_res.writelines(lines[sizeOfFold * 2:])

    if not os.path.exists(resPath + '/' + str(f)):
        os.mkdir(resPath + '/' + str(f))
        print('Successfully created directory', resPath + '/' + str(f))
    f_res = open(resPath + '/' + str(f) + '/test.txt','w',encoding='utf8')
    f_res.writelines(lines[:sizeOfFold])
    f_res = open(resPath + '/' + str(f) + '/validation.txt', 'w', encoding='utf8')
    f_res.writelines(lines[sizeOfFold:sizeOfFold * 2])
    f_res = open(resPath + '/' + str(f) + '/train.txt', 'w', encoding='utf8')
    f_res.writelines(lines[sizeOfFold * 2:])

def dataDivision811_main():
    srcPath = 'E:\Data\EmojiPrediction\emoji_sample_head_unk_delProportion.txt'
    resPath = 'E:\Data\EmojiPrediction\Fold'
    srcPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter_lower_stopwords_unk_delProportion.txt'
    resPath = conf.src_path + '/Fold'
    dataDivision811(srcPath, resPath, conf.fold)

if __name__ == "__main__":
    dataDivision811_main()
    print(sys.argv[0] + 'Finished!')