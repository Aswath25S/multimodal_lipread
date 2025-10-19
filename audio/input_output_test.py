import torch
from configs.config import load_config
from data.dataset import GLipsDataset
from torch.utils.data import DataLoader
from models.resnet_model import AudioResNet

# from models.resnet_lstm_model import AudioResNetLSTM
# from models.vgg_model import VGGAudioClassifier
# from models.vgg_lstm_model import VGGWithLSTMClassifier
# from models.lstm_resnet_model import LSTMResNet
# from models.lstm_resnet_attn_model import DeepAudioNetWithAttention
# from models.lstm_resnet_trans_model import LSTMResNetWithTransformer


def load_data(data_path, batch_size, input_size):
    train_dataset = GLipsDataset(data_path, input_size, split='train')
    val_dataset = GLipsDataset(data_path, input_size, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader

config_path = "/home/aswath/Projects/capstone/multimodel_lipread/audio/configs/audio_config.yaml"
config = load_config(config_path)
data_path = config.get('dataset.root_dir')
num_classes = config.get('dataset.num_classes')
input_size = config.get('dataset.input_size')
batch_size = config.get('training.batch_size')

train_loader, val_loader = load_data(data_path, batch_size, input_size)
temp = next(iter(train_loader))
print('Input shape:', temp[0].shape)
print('Label shape:', temp[1].shape)


inp = torch.rand(32, 80, 117)
print('Model Input shape:', inp.shape)

model1 = AudioResNet(num_classes=num_classes)
print('Model Output shape:', model1(inp).shape)

# model2 = AudioResNetLSTM(num_classes=num_classes)
# print(model2(inp).shape)

# model3 = VGGAudioClassifier(num_classes=num_classes, version=16)
# print(model3(inp).shape)

# model4 = VGGWithLSTMClassifier(num_classes=num_classes, version=19)
# print(model4(inp).shape)

# model5 = LSTMResNet(num_classes=num_classes, input_size=input_size)
# print(model5(inp).shape)

# model6 = DeepAudioNetWithAttention(num_classes=num_classes, input_size=input_size)
# print(model6(inp).shape)

# model7 = LSTMResNetWithTransformer(num_classes=num_classes, input_size=input_size)
# print(model7(inp).shape)