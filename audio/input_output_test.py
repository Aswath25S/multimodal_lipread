import torch
from configs.config import load_config
from data.dataset import GLipsDataset
from torch.utils.data import DataLoader
from models.resnet_model import AudioResNet
from models.vgg_model import VGGAudioClassifier
from models.vgg_lstm_model import VGGWithLSTMClassifier
from models.lstm_resnet_trans_model import LSTMResNetWithTransformer
from models.lstm_resnet_model import LSTMResNet
from models.lstm_resnet_attn_model import DeepAudioNetWithAttention
from models.resnet_lstm_model import AudioResNetLSTM

def get_arguments(config_path):
    config = load_config(config_path)
    data_path = config.get('dataset.root_dir')
    num_classes = config.get('dataset.num_classes')
    input_size = config.get('dataset.input_size')
    batch_size = config.get('training.batch_size')
    return config, data_path, num_classes, input_size, batch_size

def load_data(data_path, batch_size, input_size):
    train_dataset = GLipsDataset(data_path, input_size, split='train')
    val_dataset = GLipsDataset(data_path, input_size, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader

if __name__ == '__main__':
    config_path = "/home/aswath/Projects/capstone/multimodel_lipread/audio/configs/audio_config.yaml"
    config, data_path, num_classes, input_size, batch_size = get_arguments(config_path)

    train_loader, val_loader = load_data(data_path, batch_size, input_size)
    temp = next(iter(train_loader))
    print('Input shape:', temp[0].shape)
    print('Label shape:', temp[1].shape, end='\n\n')

    inp = torch.rand(32, 80, 117)
    print('Model Input shape:', inp.shape, end='\n\n')

    model1 = AudioResNet(num_classes=num_classes)
    print('Resnet Model Output shape:', model1(inp).shape, end='\n\n')

    model2 = AudioResNetLSTM(num_classes=num_classes)
    print('Resnet LSTM Model Output shape:', model2(inp).shape, end='\n\n')

    model3 = VGGAudioClassifier(num_classes=num_classes, version=16)
    print('VGG Model Output shape:', model3(inp).shape, end='\n\n')

    model4 = VGGWithLSTMClassifier(num_classes=num_classes, version=19)
    print('VGG LSTM Model Output shape:', model4(inp).shape, end='\n\n')

    model5 = LSTMResNetWithTransformer(num_classes=num_classes, input_size=input_size)
    print('LSTM ResNet Transformer Model Output shape:', model5(inp).shape, end='\n\n')

    model6 = LSTMResNet(num_classes=num_classes, input_size=input_size)
    print('LSTM ResNet Model Output shape:', model6(inp).shape, end='\n\n')

    model7 = DeepAudioNetWithAttention(num_classes=num_classes, input_size=input_size)
    print('Deep Audio Net with Attention Model Output shape:', model7(inp).shape, end='\n\n')