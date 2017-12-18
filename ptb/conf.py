# 服务器
src_path = "/home/pubsrv/data/guorui/data"
data_path = src_path + "/Fold/0"
save_path = src_path + "/Res/Fold1/0"
emb_path = src_path + "/glove.twitter.27B/glove.twitter.27B.200d.txt"
num_GPU = 1
max_max_epoch = 13#55#需要修改
hidden_size = 200#400#20#需要修改

# 本机
# src_path = "E:\Data\EmojiPrediction"
# data_path = src_path + "\Fold_head\/0"#src_path + "\Fold\/0"
# save_path = src_path + "\Res\Fold_head\/0"#src_path + "\Res\Fold\/0"
# emb_path = src_path + "\glove.twitter.27B\glove.twitter.27B.25d.txt"
# num_GPU = 0
# max_max_epoch = 1#13#55#需要修改
# hidden_size = 25#400#20#需要修改

model = "small"
rnn_mode = "BASIC"
vocab_size = 10000
init_scale = 0.1
learning_rate = 1.0
max_grad_norm = 5
num_layers = 2
num_steps = 20#默认为20，之后会变为文件中句子最大长度
max_epoch = 4#14#
keep_prob = 0.5#1.0
keep_probs = {0.5}
lr_decay = 0.5#1 / 1.15#
batch_size = 20
