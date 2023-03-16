from dataset import create_dataset
from data_preparation.create_normalization import create_dataset_norm_data

dataset_name = "BILINEAR-LED-WARIOR-scale_4-ACTUAL"

create_dataset(dataset_name, hr_dir='data/led-warior/depth_map_out/',
                                lr_dir='data/led-warior/lr_4_depth_map_out/',
                                textures_dir='data/led-warior/texture_laser/',
                                scale_lr=True,
                                def_maps=True,
                                fill=True,
                                canny=False,
                                fill_tex=True)

create_dataset_norm_data(dataset_name)