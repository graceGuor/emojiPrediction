import ptb.conf as conf
import sys

#去除只有一个Word的句子
def delSeqOfSingleWord(srcPath, resPath):
    res = []
    f_sour = open(srcPath,'r',encoding='utf8')
    lines = f_sour.readlines()
    print('before delSeqOfSingleWord size:' + str(len(lines)))
    for line in lines:
        items = line.split()
        if len(items) > 1:
            res.append(line)
    f_res = open(resPath,'w',encoding='utf8')
    print('after delSeqOfSingleWord size:' + str(len(res)))
    f_res.writelines(res)

def delSeqOfSingleWord_main():
    srcPath = 'E:\Data\EmojiPrediction\emoji_sample_head_unk_delProportion_lower.txt'
    resPath = 'E:\Data\EmojiPrediction\emoji_sample_head_unk_delProportion_lower_delProportion_singleWord.txt'
    # srcPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter_lower_stopwords_unk_delProportion.txt'
    # resPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter_lower_stopwords_unk_delProportion_singleWord.txt'
    delSeqOfSingleWord(srcPath, resPath)

if __name__ == "__main__":
    delSeqOfSingleWord_main()
    print(sys.argv[0] + 'Finished!')