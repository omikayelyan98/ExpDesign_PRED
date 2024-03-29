# <<<<<<< HEAD
from bert import modeling
from utils import load

batch_size = 16
max_history_length = 4
iter_num = 10
max_decay_iter = 4
lr = 0.00005
is_training = False

# data_path = "./twitter/twitter_attention_data/"
data_path = "./sample_data/"
text_path = "/texts.pkl"
label_path = "/labels.pkl"
ids_path = "/ids.pkl"

origin_input_idsList_path = "/input_idsList.pkl"
origin_input_masksList_path = "/input_masksList.pkl"
origin_segment_idsList_path = "/segment_idsList.pkl"
origin_one_hot_labels_path = "/one_hot_labels.pkl"

input_idsList_path = "/history_input_idsList_" + str(max_history_length) + ".pkl"
input_masksList_path = "/history_input_masksList_" + str(max_history_length) + ".pkl"
segment_idsList_path = "/history_segment_idsList_" + str(max_history_length) + ".pkl"
one_hot_labels_path = "/history_one_hot_labels_" + str(max_history_length) + ".pkl"

train_data_type = "train"
test_data_type = "test"
val_data_type = "val"

# 这里是下载下来的bert配置文件, 未 fine tuning的Bert path
config_path = "./uncased_L-8_H-512_A-8/"
init_checkpoint = config_path + "bert_model.ckpt"

# fine-tuning  path
# config_path = "./fine_tuning/"
# init_checkpoint = config_path + "model.ckpt-396000"

bert_config = modeling.BertConfig.from_json_file(config_path + "bert_config.json")
vocab_file = config_path + "vocab.txt"

# num_labels = len(load(data_path + "train/label_dic.pkl"))
num_labels = len(load(data_path + "train/labels.pkl"))

max_seq_length = 128
if max_seq_length > bert_config.max_position_embeddings:
    raise ValueError("超出模型最大长度")
"""# =======
from bert import modeling
from utils import load

batch_size = 16
max_history_length = 4
iter_num = 10
max_decay_iter = 4
lr = 0.00005
is_training = False

# data_path = "./twitter/twitter_attention_data/"
data_path = "./sample_data/"
text_path = "/texts.pkl"
label_path = "/labels.pkl"
ids_path = "/ids.pkl"

origin_input_idsList_path = "/input_idsList.pkl"
origin_input_masksList_path = "/input_masksList.pkl"
origin_segment_idsList_path = "/segment_idsList.pkl"
origin_one_hot_labels_path = "/one_hot_labels.pkl"

input_idsList_path = "/history_input_idsList_" + str(max_history_length) + ".pkl"
input_masksList_path = "/history_input_masksList_" + str(max_history_length) + ".pkl"
segment_idsList_path = "/history_segment_idsList_" + str(max_history_length) + ".pkl"
one_hot_labels_path = "/history_one_hot_labels_" + str(max_history_length) + ".pkl"

train_data_type = "train"
test_data_type = "test"
val_data_type = "val"

# 这里是下载下来的bert配置文件, 未 fine tuning的Bert path
config_path = "./uncased_L-8_H-512_A-8/"
init_checkpoint = config_path + "bert_model.ckpt"

# fine-tuning  path
# config_path = "./fine_tuning/"
# init_checkpoint = config_path + "model.ckpt-396000"

bert_config = modeling.BertConfig.from_json_file(config_path + "bert_config.json")
vocab_file = config_path + "vocab.txt"


num_labels = len(load(data_path + "train/label_dic.pkl"))

max_seq_length = 128
if max_seq_length > bert_config.max_position_embeddings:
    raise ValueError("超出模型最大长度")
# >>>>>>> fad0c2c7d5f1bd28b868c1169bc7a25da0b518c5
"""