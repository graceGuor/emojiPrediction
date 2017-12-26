import ptb.conf as conf

#将大写字母小写化
def lowerCha(srcPath, resPath):
    res = []
    f_sour = open(srcPath,'r',encoding='utf8')
    lines = f_sour.readlines()
    for line in lines:
        line = line.lower()
        res.append(line)
    f_res = open(resPath,'w',encoding='utf8')
    f_res.writelines(res)

if __name__ == "__main__":
    srcPath = 'E:\Data\EmojiPrediction\emoji_sample_head_unk_delProportion.txt'
    resPath = 'E:\Data\EmojiPrediction\emoji_sample_head_unk_delProportion_lower.txt'

    srcPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter_stopwords_unk_delProportion.txt'
    resPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter_stopwords_unk_delProportion_lower.txt'

    lowerCha(srcPath,resPath)
    print('Finished!')