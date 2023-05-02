from dataset import create_dataset
from data_preparation.create_normalization import create_dataset_norm_data

dataset_name = "NEAREST-INFERENCE-TEST"

create_dataset(dataset_name, hr_dir='data/inference/depth_map_out/',
                                lr_dir='data/inference/lr_4_depth_map_out/',
                                textures_dir='data/inference/texture_laser/',
                                scale_lr=True,
                                def_maps=True,
                                fill=True,
                                canny=False,
                                fill_tex=True)

create_dataset_norm_data(dataset_name)