# **Multi-label ECG Abnormality Classification Using A Combined ResNet-DenseNet Architecture with ResU Blocks**   
   
This is an official repo of the paper "**Multi-label ECG Abnormality Classification Using A Combined ResNet-DenseNet Architecture with ResU Blocks**," which is submitted to IEEE EMBS International Conference on Data Science and Engineering in Healthcare, Medicine & Biology.   

**Abstract**：Electrocardiogram (ECG) abnormality classification is to detect various types of clinical abnormalities from ECG. 
This paper proposes a novel Deep Neural Network (DNN)-based ECG abnormality classification based on an architecture where ResNet and DenseNet are cascaded.
ResNet in the proposed architecture comprises a residual U-shaped (ResU) block that effectively captures multi-scale feature maps without significantly increasing parameters.
In addition, we use a multi-head self-attention (MHSA) to ensure that the model focuses better on clinically essential features in the given ECG.
Experimental results show that our proposed model performs superior ECG abnormality classification performance to other recently proposed DNN-based models.

## Update:  
* **2023.09.20** Upload codes  

## Requirements 
This repo is tested with Ubuntu 22.04, PyTorch 2.0.1, Python3.10, and CUDA11.7. For package dependencies, you can install them by:

```
pip install -r requirements.txt    
```   


## Getting started    
1. Install the necessary libraries.   
2. Download the PhysioNet Challenge 2021 database and place it in '../Dataset/' folder.   
```
├── 📦 ResU_Dense   
│   └── 📂 dataset   
│       └── 📜 train_dataset.csv   
│       └── 📜 test_dataset.csv   
│   └── ...   
└── 📦 Dataset   
    └── 📂 physionet_challenge_dataset
        └── 📂 physionet.org 
            └── ...
```

3. Run [train_interface.py](https://github.com/seorim0/ResU_Dense/blob/main/train_interface.py)
  * You can simply change any parameter settings if you need to adjust them.   ([options.py](https://github.com/seorim0/ResU_Dense/blob/main/options.py)) 


## Results  
![Screenshot from 2023-09-20 21-11-12](https://github.com/seorim0/ResU_Dense/assets/55497506/cda05b47-2993-4033-97a7-c0be0d0ce9c6)  

    
![f1score_per_class_230914](https://github.com/seorim0/ResU_Dense/assets/55497506/ee3f2247-6293-4170-ad86-e18471094ccf)  


## Reference   
**Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021**    
Matthew Reyna, Nadi Sadr, Annie Gu, Erick Andres Perez Alday, Chengyu Liu, Salman Seyedi, Amit Shah, and Gari Clifford  
[[paper]](https://physionet.org/content/challenge-2021/1.0.3/)   
**Automatic diagnosis of the 12-lead ECG usinga deep neural network**    
Antônio H. Ribeiro, et al.  
[[paper]](https://www.nature.com/articles/s41467-020-15432-4) [[code]](https://github.com/antonior92/automatic-ecg-diagnosis)  
**A multi-view multi-scale neural network for multi-label ECG classification**    
Shunxiang Yang, Cheng Lian, Zhigang Zeng, Bingrong Xu, Junbin Zang, and Zhidong Zhang  
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10021962) [[code]](https://github.com/ysxGitHub/MVMS-net)    
**Classification of ECG using ensemble of residual CNNs with attention mechanism**    
Petr Nejedly, Adam Ivora, Radovan Smisek, Ivo Viscor, Zuzana Koscova, Pavel Jurak, and Filip Plesinger  
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9662723) [[code]](https://moody-challenge.physionet.org/2021/)  


## Contact  
Please get in touch with us if you have any questions or suggestions.   
E-mail: allmindfine@yonsei.ac.kr (Seorim Hwang) / jbcha7@yonsei.ac.kr (Jaebine Cha)
