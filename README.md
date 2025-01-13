# DSTSA-GCN_Gesture

PyTorch implementation of “DSTSA-GCN: Advancing Skeleton-Based Gesture Recognition with Semantic-Aware Spatio-Temporal Topology Modeling”.

[DSTSA-GCN : Proj](https://hucui2022.github.io/dstsa_gcn/)

# Data Preparation

### method 1:

- SHREC
  - Download the SHREC data from http://www-rech.telecom-lille.fr/shrec2017-hand/
- DHG
  - Download the DHG data from the http://www-rech.telecom-lille.fr/DHGdataset/
- NTU-60
  - Download the NTU-60 data from the [ROSE Lab](https://rose1.ntu.edu.sg/dataset/actionRecognition/) or https://drive.google.com/open?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H
- NTU-120
  - Download the NTU-120 data from the  [ROSE Lab](https://rose1.ntu.edu.sg/dataset/actionRecognition/) or https://drive.google.com/open?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H
- Note 1
  - NTU 60 and 120: 
    
    step 1 : run :  "get_raw_skes_data.py" extract raw skeleton data.  
    
    step 2 : run :   "get_raw_denoised_data.py" remove denoised frames.  
    
    step 3: run:  "seq_transformation.py" get   xxx.npz datasets.  
    
    then you can play fun with ntu dataset.   Maybe need change some path (like: 'E:/DataSets/sttf_ntu/ntu60/') in the code by yourself for your system. 
  
  - SHREC and DHG :  need to change datapth in feeder.py 

### method 2:  download from clod drive

- SHREC'17 : Download from [Google Drive](https://drive.google.com/file/d/1lhbbR22QcJWGT4NpOvypqx-euQ6bkwVd/view?usp=sharing).

- DHG : Download from [Google Drive](https://drive.google.com/file/d/1GIM3qQRrfHzZbRusXpcrakWQR2n31t86/view?usp=sharing).

- **NTU RGB+D 60** dataset from [Baidu Drive](https://pan.baidu.com/s/16WmFFkGwZM6be93L376WUQ?pwd=TDGC) 

## Training :

### Shrec

```
python main.py --config configs/shrec17/14/j.yaml 
```

### DHG

```
python main.py --config  configs/dgh/dgh14/joint.yaml
```

### NTU-60

```
python main.py --config configs/nut60/xsub/joint.yaml
```

### NTU-120

```
python main.py --config configs/nut120/xsub/joint.yaml
```



## Testing

```
python main.py --config configs/nut60/xsub/joint.yaml --phase test --weights xxxx
```

## Ensemble

```
python ensemble.py --config ensemble.yaml
```



## Citation

```
@article{cui2024joint,
  title={Joint-Partition Group Attention for skeleton-based action recognition},
  author={Cui, Hu and Hayama, Tessai},
  journal={Signal Processing},
  volume={224},
  pages={109592},
  year={2024},
  publisher={Elsevier}
}
```

Our project is based on the :  [DSTA-Net](https://github.com/lshiwjx/DSTA-Net), [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN),[DG-STGCN](https://github.com/kennymckormick/pyskl/blob/main/configs/dgstgcn/README.md) [TD-GCN](https://github.com/liujf69/TD-GCN-Gesture)
