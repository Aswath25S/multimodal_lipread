from dataset_loader import get_data_loaders

config_path = '/home/aswath/Projects/capstone/multimodel_lipread/video/config/visual_config.yaml'
train_loader, val_loader, test_loader = get_data_loaders(config_path)

print(train_loader.__len__())
print(val_loader.__len__())
print(test_loader.__len__())

for batch in train_loader:
    print(batch['lip_regions'].shape)
    print(batch['label'].shape)
    break
