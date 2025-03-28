# DSTSA-GCN_Gesture

PyTorch implementation of “DSTSA-GCN: Advancing Skeleton-Based Gesture Recognition with Semantic-Aware Spatio-Temporal Topology Modeling”.

[DSTSA-GCN : Proj](https://hucui2022.github.io/dstsa_gcn/)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dstsa-gcn-advancing-skeleton-based-gesture/hand-gesture-recognition-on-dhg-28)](https://paperswithcode.com/sota/hand-gesture-recognition-on-dhg-28?p=dstsa-gcn-advancing-skeleton-based-gesture)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dstsa-gcn-advancing-skeleton-based-gesture/skeleton-based-action-recognition-on-shrec)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-shrec?p=dstsa-gcn-advancing-skeleton-based-gesture)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dstsa-gcn-advancing-skeleton-based-gesture/hand-gesture-recognition-on-dhg-14)](https://paperswithcode.com/sota/hand-gesture-recognition-on-dhg-14?p=dstsa-gcn-advancing-skeleton-based-gesture)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dstsa-gcn-advancing-skeleton-based-gesture/action-recognition-in-videos-on-ntu-rgbd-120)](https://paperswithcode.com/sota/action-recognition-in-videos-on-ntu-rgbd-120?p=dstsa-gcn-advancing-skeleton-based-gesture)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dstsa-gcn-advancing-skeleton-based-gesture/skeleton-based-action-recognition-on-n-ucla)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-n-ucla?p=dstsa-gcn-advancing-skeleton-based-gesture)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dstsa-gcn-advancing-skeleton-based-gesture/action-recognition-in-videos-on-ntu-rgbd)](https://paperswithcode.com/sota/action-recognition-in-videos-on-ntu-rgbd?p=dstsa-gcn-advancing-skeleton-based-gesture)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dstsa-gcn-advancing-skeleton-based-gesture/skeleton-based-action-recognition-on-ntu-rgbd-1)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd-1?p=dstsa-gcn-advancing-skeleton-based-gesture)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dstsa-gcn-advancing-skeleton-based-gesture/skeleton-based-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd?p=dstsa-gcn-advancing-skeleton-based-gesture)


# Data Preparation

### method 1:

- SHREC
  - Download the SHREC data from http://www-rech.telecom-lille.fr/shrec2017-hand/
  - Generate the train/test splits with `python prepare/shrec/gendata.py`
- DHG
  - Download the DHG data from the http://www-rech.telecom-lille.fr/DHGdataset/
  - Generate the train/test splits with `python prepare/dhg/gendata.py`
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
- UCLA dataset from : [Google Drive](https://www.dropbox.com/scl/fi/6numm9wzu1cixw8nyzb91/all_sqe.zip?rlkey=it1ruxtsm4rggxldbbbr4w3yj&e=1&dl=0)

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

### NW-UCLA

```
python main.py --config configs/ucla/j.yaml
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
@article{CUI2025130066,
title = {Dstsa-gcn: Advancing skeleton-based gesture recognition with semantic-aware spatio-temporal topology modeling},
journal = {Neurocomputing},
pages = {130066},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2025.130066},
url = {https://www.sciencedirect.com/science/article/pii/S0925231225007386},
author = {Hu Cui and Renjing Huang and Ruoyu Zhang and Tessai Hayama}
}
```

Our project is based on the :  [DSTA-Net](https://github.com/lshiwjx/DSTA-Net), [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN),[DG-STGCN](https://github.com/kennymckormick/pyskl/blob/main/configs/dgstgcn/README.md) [TD-GCN](https://github.com/liujf69/TD-GCN-Gesture)
