# SRCNN
- Name: GunGyeom James Kim
- OS: Windows 11 / Colab(T4 GPU)

You can just run train and test on Colab using *colab.ipynb* or on your terminal by following below requiement and commands

## Requirements
- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0

## Train
The 91-image, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| 91-image | 2 | Train | [Download](https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=0) |
| 91-image | 3 | Train | [Download](https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0) |
| 91-image | 4 | Train | [Download](https://www.dropbox.com/s/22afykv4amfxeio/91-image_x4.h5?dl=0) |
| Set5 | 2 | Eval | [Download](https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=0) |
| Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=0) |
| Set5 | 4 | Eval | [Download](https://www.dropbox.com/s/0rz86yn3nnrodlb/Set5_x4.h5?dl=0) |


```bash
python train.py --train-file "data/91-image_x3.h5" \
                --eval-file "data/Set5_x3.h5" \
                --outputs-dir "pth" \
                --scale 3 \
                --lr 1e-4 \
                --batch-size 16 \
                --num-epochs 400 \
                --seed 123                
```

## Test
The 91-image, Set5 dataset of *png* type can be downloaded from [here](https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU?usp=sharing).

```bash
python test.py --weights-file "pth/srcnn_x3.pth" \
               --image-file "data/butterfly.png" \
               --scale 3
```
