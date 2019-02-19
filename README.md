# PiCANet-Implementation
Pytorch Implementation of [**PiCANet: Learning Pixel-wise Contextual Attention for Saliency Detection**](https://arxiv.org/abs/1708.06433)


## New method on implementing PiCANet
* [Issue#9](https://github.com/Ugness/PiCANet-Implementation/issues/9)
* Conv3d version is deleted.

![input image](readme_images/input.png)
![target_image](readme_images/mask.png)
* batchsize 1
![training_result](readme_images/Training_result.JPG)  
* batchsize 4
![training_result](readme_images/Training_result_new.JPG)  

# Execution Guideline
## Requirements
Pillow==4.3.0  
pytorch==0.4.1  
tensorboardX==1.1  
torchvision==0.2.1  
numpy==1.14.2  

## My Environment
S/W  
Windows 10  
CUDA 9.0  
cudnn 7.0  
python 3.5  
H/W  
AMD Ryzen 1700  
Nvidia gtx 1080ti  
32GB RAM

## Execution Guide
- For training,
- Please check the Detailed Guideline if you want to know the [dataset](#pairdataset-class) structure.

<pre>
    usage: train.py [-h] [--load LOAD] --dataset DATASET [--cuda CUDA]
                    [--batch_size BATCH_SIZE] [--epoch EPOCH] [-lr LEARNING_RATE]
                    [--lr_decay LR_DECAY] [--decay_step DECAY_STEP]
                    [--display_freq DISPLAY_FREQ]
    optional arguments:
      -h, --help            show this help message and exit
      --load LOAD           Directory of pre-trained model, you can download at
                            https://drive.google.com/file/d/109a0hLftRZ5at5hwpteRfO1A6xLzf8Na/view?usp=sharing
                            None --> Do not use pre-trained model. Training will start from random initialized model
      --dataset DATASET     Directory of your Dataset
      --cuda CUDA           'cuda' for cuda, 'cpu' for cpu, default = cuda
      --batch_size BATCH_SIZE
                            batchsize, default = 1
      --epoch EPOCH         # of epochs. default = 20
      -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                            learning_rate. default = 0.001
      --lr_decay LR_DECAY   Learning rate decrease by lr_decay time per decay_step, default = 0.1
      --decay_step DECAY_STEP
                            Learning rate decrease by lr_decay time per decay_step,  default = 7000
      --display_freq DISPLAY_FREQ
                            display_freq to display result image on Tensorboard

</pre>

- For inference,
- dataset should contain image files only.
- You do not need `masks` or `images` folder. If you want to run with PairDataset structure, use argument like  
```--dataset [DATAROOT]/images```
- You should specify either logdir (for TensorBoard output) or save_dir (for Image file output).
- If you use logdir, you can see the whole images by run tensorboard with  `--samples_per_plugin images=0` option

<pre>
    usage: image_test.py [-h] [--model_dir MODEL_DIR] --dataset DATASET
                     [--cuda CUDA] [--batch_size BATCH_SIZE] [--logdir LOGDIR]
                     [--save_dir SAVE_DIR]

    optional arguments:
      -h, --help            show this help message and exit
      --model_dir MODEL_DIR
                            Directory of pre-trained model, you can download at
                            https://drive.google.com/drive/folders/1s4M-_SnCPMj_2rsMkSy3pLnLQcgRakAe?usp=sharing
      --dataset DATASET     Directory of your test_image ""folder""
      --cuda CUDA           cuda for cuda, cpu for cpu, default = cuda
      --batch_size BATCH_SIZE
                            batchsize, default = 4
      --logdir LOGDIR       logdir, log on tensorboard
      --save_dir SAVE_DIR   save result images as .jpg file. If None -> Not save

</pre>

- To report score,
- dataset should contain `masks` and `images` folder.
- You should specify logdir to get PR-Curve.
- The Scores will be printed out on your stdout.
- You should have **model files** below the model_dir.
- Only supports model files named like **"[N]epo_[M]step.ckpt"** format.
<pre>
    usage: measure_test.py [-h] --model_dir MODEL_DIR --dataset DATASET
                       [--cuda CUDA] [--batch_size BATCH_SIZE]
                       [--logdir LOGDIR] [--which_iter WHICH_ITER]
                       [--cont CONT] [--step STEP]

    optional arguments:
      -h, --help            show this help message and exit
      --model_dir MODEL_DIR
                            Directory of folder which contains pre-trained models, you can download at
                            https://drive.google.com/drive/folders/1s4M-_SnCPMj_2rsMkSy3pLnLQcgRakAe?usp=sharing
      --dataset DATASET     Directory of your test_image ""folder""
      --cuda CUDA           cuda for cuda, cpu for cpu, default = cuda
      --batch_size BATCH_SIZE
                            batchsize, default = 4
      --logdir LOGDIR       logdir, log on tensorboard
      --which_iter WHICH_ITER
                            Specific Iter to measure
      --cont CONT           Measure scores from this iter
      --step STEP           Measure scores per this iter step

</pre>


# Detailed Guideline
### Pretrained Model
You can download pre-trained models from https://drive.google.com/drive/folders/1s4M-_SnCPMj_2rsMkSy3pLnLQcgRakAe?usp=sharing  
## Dataset
### PairDataset Class
* You can use CustomDataset.
* Your custom dataset should contain `images`, `masks` folder.
  - In each folder, the filenames should be matched. 
  - eg. ```images/a.jpg masks/a.jpg```
### DUTS
You can download dataset from http://saliencydetection.net/duts/#outline-container-orgab269ec.
* Caution: You should check the dataset's Image and GT are matched or not. (ex. # of images, name, ...)
* You can match the file names and automatically remove un-matched datas by using `DUTSDataset.arrange(self)` method
* Please rename the folders to `images` and `masks`.

### Directory & Name Format of .ckpt files
<code>
        "models/state_dict/<datetime(Month,Date,Hour,Minute)>/<#epo_#step>.ckpt"
</code>

* The step is accumulated step from epoch 0.
