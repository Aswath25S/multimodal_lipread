from config.config import load_config
from data_utils.dataset_av import GLipsMultimodalDataset
from models.ef_cnn_lstm_resnet import create_early_fusion_model
from torch.utils.data import DataLoader


config_path = "/home/aswath/Projects/capstone/multimodel_lipread/audio_video/config/av_config.yaml"
config = load_config(config_path)

dataset = GLipsMultimodalDataset(root_dir=config.get("dataset.root_dir"), input_size_audio=config.get("dataset.audio_input_size"), split='train')
dataloader = DataLoader(dataset, batch_size=config.get('training.batch_size'), shuffle=True)

model = create_early_fusion_model(num_classes=config.get("dataset.num_classes"), config=config)


for batch in dataloader:
    audio, video, label = batch
    print(audio.shape, video.shape)
    print(model(audio, video))
    break

