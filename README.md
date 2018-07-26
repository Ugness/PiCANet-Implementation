# PiCANet-Implementation
Pytorch Implementation of [**PiCANet: Learning Pixel-wise Contextual Attention for Saliency Detection**](https://arxiv.org/abs/1708.06433)

![input image](/input.png)
![target_image](/mask.png)

* You can execute train.py by Download DUTS Dataset and modify the _root_dir_ argument in DUTS_Dataset(), train.py

# Performance Test with F-score (beta-square = 0.3)  
| Step   | Value              |
|--------|--------------------|
| 10000  | 0.7106643319129944 |
| 20000  | 0.7546798586845398 |
| 30000  | 0.7427917718887329 |
| 40000  | 0.7690391540527344 |
| 50000  | 0.7736681699752808 |
| 60000  | 0.7714764475822449 |
| 70000  | 0.7833986282348633 |
| 80000  | 0.7559545040130615 |
| 90000  | 0.7594088315963745 |
| 110000 | 0.7517607808113098 |
| 120000 | 0.6820743680000305 |
| 130000 | 0.7404838800430298 |
| 140000 | 0.7366619110107422 |
| 150000 | 0.7543904781341553 |
| 160000 | 0.7171811461448669 |
| 170000 | 0.7377526760101318 |
| 180000 | 0.7234616875648499 |
| 190000 | 0.7803277373313904 |
| 200000 | 0.7252792716026306 |
| 210000 | 0.7865139245986938 |

# Execution Guideline
## Requirements
Pillow==4.3.0  
pytorch==0.4.0  
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

## You can run the file by following the descriptions in -h option.
<code>
    python train.py -h
</code>
<pre>
    usage: train.py [-h] [--load LOAD] [--dataset DATASET] [--cuda CUDA]
                [--batch_size BATCH_SIZE] [--epoch EPOCH] [-lr LEARNING_RATE]
                [--lr_decay LR_DECAY] [--decay_step DECAY_STEP]
    
    optional arguments:
    -h, --help            show this help message and exit
    --load LOAD           Directory of pre-trained model, you can download at 
                        https://drive.google.com/file/d/109a0hLftRZ5at5hwpteRfO1
                        A6xLzf8Na/view?usp=sharing
                        None --> Do not use pre-trained model. 
                        Training will start from random initialized model
    --dataset DATASET     Directory of your DUTS dataset "folder"
    --cuda CUDA           'cuda' for cuda, 'cpu' for cpu, default = cuda
    --batch_size BATCH_SIZE
                        batchsize, default = 1
    --epoch EPOCH         # of epochs. default = 20
    -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning_rate. default = 0.001
    --lr_decay LR_DECAY   Learning rate decrease by lr_decay time per decay_step, default = 0.1
    --decay_step DECAY_STEP
                        Learning rate decrease by lr_decay time per decay_step, default = 7000
</pre>

<code>
    python Image_Test.py -h
</code>
<pre>
    usage: Image_Test.py [-h] [--model_dir MODEL_DIR] [-img IMAGE_DIR]
                         [--cuda CUDA] [--batch_size BATCH_SIZE]

    optional arguments:
      -h, --help            show this help message and exit
      --model_dir MODEL_DIR
                            Directory of pre-trained model, you can download at
                            https://drive.google.com/drive/folders/1s4M-_SnCPMj_2rsMkSy3pLnLQcgRakAe?usp=sharing
      -img IMAGE_DIR, --image_dir IMAGE_DIR
                            Directory of your test_image ""folder""
      --cuda CUDA           'cuda' for cuda, 'cpu' for cpu, default = cuda
      --batch_size BATCH_SIZE
                            batchsize, default = 4
</pre>

## Detailed Guideline
### Pretrained Model
You can download pre-trained models from https://drive.google.com/drive/folders/1s4M-_SnCPMj_2rsMkSy3pLnLQcgRakAe?usp=sharing  
### Dataset
I used DUTS dataset as Training dataset and Test dataset.  
You can download dataset from http://saliencydetection.net/duts/#outline-container-orgab269ec.
* Caution: You should check the dataset's Image and GT are matched or not. (ex. # of images, name, ...)

### Execution Example
Assume you train the model with  
* current dir: Pytorch/  
* Dataset dir: Pytorch/DUTS-TE  
* Pretrained model dir: Pytorch/models/state_dict/07261950/10epo_1000000step.ckpt  
* Goal Epoch : 100  
<code>
    python train.py --load models/state_dict/07261950/10epo_1000000step.ckpt --dataset DUTS-TE --epoch 100
</code>

Assume you test the model with  
* current dir: Pytorch/  
* Testset dir: Pytorch/test  
* Pretrained model dir: Pytorch/models/state_dict/07261950/10epo_1000000step.ckpt  
* CPU mode  
<code>
    python Image_test.py --model_dir models/state_dict/07261950/10epo_1000000step.ckpt --img test --cuda cpu
</code>

### Directory & Name Format of .ckpt files
<code>
        "models/state_dict/<datetime(Month,Date,Hour,Minute)>/<#epo_#step>.ckpt"
</code>

* The step is accumulated step from epoch 0.
* If you want to change the format of pre-trained model, you should change the code in train.py line 61-65
    ```
    start_iter = int(load.split('epo_')[1].strip('step.ckpt')) + 1
    start_epo = int(load.split('/')[3].split('epo')[0])
    now = datetime.datetime.strptime(load.split('/')[2], '%m%d%H%M')
    ```

### Test with Custom_Images
* When you run Image_Test.py with your own Images, the images will saved in tensorboard log file.

* Log files will saved in log/Image_test

* You can see the images by execute
    ```
    tensorboard --logdir=./log/Image_test
    ```

    and browse 127.0.0.1:6006 with your browser(ex. Chrome, IE, etc)
