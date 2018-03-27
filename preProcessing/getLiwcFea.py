import ptb.conf as conf
import sys
import tensorflow as tf
import Service.ReadInfo as RI
import os

#得到LIWC的特征，每句话到当前词的各种类别词的计数统计
def getLiwcFea(srcPath, liwcPath, resPath):
    liwc_fea_dict = RI.loadDict_csv(liwcPath)
    f = tf.gfile.GFile(srcPath, "r")
    data = f.read().replace("\n", " <eos> ").split()
    # print(data[len(data) - 1])# data最后一个Word为<eos>
    fea_dim = len(liwc_fea_dict.get("\\ufeffWord"))#第一行为类别，特征维度
    count = [0] *  fea_dim# 每句话当前所有类别之和计数
    res = []
    # res = [count] * len(data)  # data个数 * 特征维度的0矩阵，太大了
    firstLine = data[0] + ', ' + str(count).replace('[', '').replace(']', '') + '\n'
    res.append(firstLine)

    for i in range(len(data) - 1):
        if data[i] in liwc_fea_dict:
            count = [m + n for m, n in zip(count, liwc_fea_dict[data[i]])]
        else:
            count = count # liwc词典中没有该词，则该词置为0
        res_item = data[i + 1] + ', ' + str(count).replace('[', '').replace(']', '') + '\n'
        res.append(res_item)
        if data[i] == '<eos>':
            count = [0] * fea_dim  # 置为0，每句话的开始

    f_res = open(resPath, 'w', encoding='utf8')
    print('word size:' + str(len(res)))
    f_res.writelines(res)

def getLiwcFea_main():
    srcPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter_lower_stopwords_unk_delProportion_singleWord.txt'
    liwcPath = os.path.join(conf.src_path, 'LIWC2015 Results (emoji_sample_withBlankbeforePunc_blankEmo_merge_filter_lower_stopwords_unk_delProportion_singleWord).csv')
    resPath = conf.src_path + '/LIWC2015 Results (emoji_sample)Word category_fea.txt'
    getLiwcFea(srcPath, liwcPath, resPath)

if __name__ == "__main__":
    getLiwcFea_main()
    print(sys.argv[0] + ' Finished!')