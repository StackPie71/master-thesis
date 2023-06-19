# Multimodal segmentation of head and neck cancer GTV from CT & MRI images

This repository proivdes source code for multimodal segmentation of Gross Target Volume (GTV) of head and neck cancer from CT and MRI images.

The code has been greatly inspired by this paper:

* [1] Haochen Mei, Wenhui Lei, Ran Gu, Shan Ye, Zhengwentai Sun, Shichuan Zhang and Guotai Wang. "Automatic Segmentation of Gross Target Volume of Nasopharynx Cancer using Ensemble of Multiscale Deep Neural Networks with Spatial Attention." NeuroComputing, accepted. 2020.

# Requirement
* Pytorch version >=0.4.1
* Some common python packages such as Numpy, Pandas, SimpleITK


## Data and preprocessing
1. Create two folders in your saveroot, like `saveroot/data` and `saveroot/label`. Then set `dataroot` and  `saveroot` and then run `python movefiles.py` in Data_preprocessing folder to save the images and annotations to a single folder respectively.
2. Create three folders for each scale image in your saveroot and then create two folders in each of them like `saveroot/small_sacle/data` and `saveroot/small_sacle/label`. Run `python preprocess.py` in Data_preprocessing folder to perform preprocessing for the images and annotations and then save each of them to a single folder respectively.
3. Set `saveroot` according to your computer in `examples/miccai/write_csv_files.py` and run `python write_csv_files.py` to randomly split the 20 images into training (16) testing (4) sets. The validation set and testing set are the same in our experimental setting. The output csv files are saved in `config`.

## Training
1. Set the value of `root_dir` as your `GTV_root` in `config/train_test.cfg`. Add the path of `PyMIC` to `PYTHONPATH` environment variable (if you haven't done this). Then you can start trainning by running following command:
 
```bash
export PYTHONPATH=$PYTHONPATH:your_path_of_PyMIC
python ../../pymic/train_infer/train_infer.py train config/train_test.cfg
```

2. During training or after training, run the command `tensorboard --logdir model/2D5unet` and then you will see a link in the output, such as `http://your-computer:6006`. Open the link in the browser and you can observe the average Dice score and loss during the training stage. 

## Testing and evaluation
1. After training, run the following command to obtain segmentation results of your testing images:

```bash
mkdir result
python ../../pymic/train_infer/train_infer.py test config/train_test.cfg
```

2. Then replace `ground_truth_folder` with your own `GTV_root/label` in `config/evaluation.cfg`, and run the following command to obtain quantitative evaluation results in terms of dice. 

```bash
python ../../pymic/util/evaluation.py config/evaluation.cfg
```
