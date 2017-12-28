import ptb.conf as conf
import sys

#去除unk比例高于一定值的句子
def delUnkProportion(srcPath, resPath):
    res = []
    f_sour = open(srcPath,'r',encoding='utf8')
    lines = f_sour.readlines()
    for line in lines:
        items = line.split()
        if len(items) == 0:
            continue
        count = 0
        for item in items:
            if item == '<unk>':
                count += 1
        proportion = count / len(items)
        if proportion <= 1 - conf.unkProportion:
            res.append(line)
    f_res = open(resPath,'w',encoding='utf8')
    print('after delUnkProportion size:' + str(len(res)))
    f_res.writelines(res)

def delUnkProportion_main():
    srcPath = 'E:\Data\EmojiPrediction\emoji_sample_head_unk.txt'
    resPath = 'E:\Data\EmojiPrediction\emoji_sample_head_unk_delProportion.txt'
    srcPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter_lower_stopwords_unk.txt'
    resPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter_lower_stopwords_unk_delProportion.txt'
    delUnkProportion(srcPath, resPath)

if __name__ == "__main__":
    delUnkProportion_main()
    print(sys.argv[0] + 'Finished!')