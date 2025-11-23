from data_utils.dataset_loader import get_data_loaders
from models.resnet_lstm import create_model
from config.config import load_config

config_path = '/home/aswath/Projects/capstone/multimodel_lipread/video/config/visual_config.yaml'
config = load_config(config_path)

train_loader, val_loader, test_loader = get_data_loaders(config_path)
model = create_model(num_classes=len(train_loader.dataset.classes), config=config)

print(train_loader.__len__())
print(val_loader.__len__())
print(test_loader.__len__())

for batch in train_loader:
    print(batch['lip_regions'].shape)
    print(batch['label'].shape)
    print(model(batch['lip_regions']))
    break
