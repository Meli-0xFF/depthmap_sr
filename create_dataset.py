from dataset import create_dataset
from data_preparation.create_normalization import create_dataset_norm_data

dataset_name = "warior-scale_4-filled-with_canny"

create_dataset(dataset_name, hr_dir='data/led-warior/depth_map_out/',
                                lr_dir='data/led-warior/lr_4_depth_map_out/',
                                textures_dir='data/led-warior/texture_laser/',
                                scale_lr=True,
                                def_maps=True,
                                fill=True,
                                canny=True)

create_dataset_norm_data(dataset_name)