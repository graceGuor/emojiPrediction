import ptb.conf as conf
import sys

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

def lowerCha_main():
    srcPath = 'E:\Data\EmojiPrediction\emoji_sample_head_unk_delProportion.txt'
    resPath = 'E:\Data\EmojiPrediction\emoji_sample_head_unk_delProportion_lower.txt'
    srcPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter.txt'
    resPath = conf.src_path + '/emoji_sample_withBlankbeforePunc_blankEmo_merge_filter_lower.txt'
    lowerCha(srcPath, resPath)

if __name__ == "__main__":
    lowerCha_main()
    print(sys.argv[0] + 'Finished!')