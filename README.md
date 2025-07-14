# Early Stroke Diagnosis and Evaluation Based on Pathological Voice Classification Using Speech Enhancement
This paper proposes a cascaded framework for early stroke diagnosis (ESD) combining speech enhancement (based on SEWUNet) and pathological voice recognition. It processes sustained vowels (SVs) with handcrafted features and machine learning algorithms, and spontaneous speech (SS) with mel-spectrogram features and a CNN-Transformer model. Clinical trials show that the two-stage strategy (SVs followed by SS) achieves 100% accuracy, demonstrating its potential as an assistive diagnostic tool.
## Requirements
- Python==3.8.13
- torch==1.12.1
- numpy==1.23.5
- scikit-learn==1.1.2
- pillow==10.4.0
## Project structure
```
ESPVC/
├── model_ss_sv/
│ ├── ss_train.py 
│ ├── sv_train.ipynb
│ ├── pytorchtools.py
│ ├── SerialNet.py
│ ├── sparse_image_warp_pytorch.py
│ ├── spec_augment_pytorch.py
├── SEWUNet/
│ ├── autoencoder_SEWUNet.ipynb
│ ├── denoising_SEWUNet.ipynb
│ ├── parts.py
│ ├── utils.py
```
## Training
```
python ss_train.py
python sv_train.py
denoising_SEWUNet.py
```
# Results
1.The performance of the proposed AdaBoost model in SVs recognition.
| Voice Type | Model |  Accuracy | Sensitivity | Specificity| F1-Score|
| :---------: | :----: | :--------: | :----------: | :---------: | :------: |
|     SVs    |AdaBoost|  92.31%  |    -        | -          | -       |

2.The performance of the proposed CNN-Transformer model in SS recognition.
| Voice Type | Model |  Accuracy | Sensitivity | Specificity| F1-Score|
| :---------: | :----: | :--------: | :----------: | :---------: | :------: |
|     SS     |	CNN-Transformer|  95.00%  |   95.37%    |   93.75%     | 95.37%    |
|     SS     |ResNet-Transformer| 96.00%  |    96.30%   | 	94.79%     | 96.30%    |

3.The evaluation of stroke recognition models before and after enhancement.
| Voice Type | Model |  Accuracy | Sensitivity | Specificity| F1-Score|
| :---------: | :----: | :--------: | :----------: | :---------: | :------: |
|     Denoised SVs    |	AdaBoost|  91.97%  |   95.00%    |   86.75%    | 93.74%   |
|     Denoised SS     |CNN-Transformer| 94.12%  |   96.30%     | 	91.67%    | 94.52%   |

4.Test results of the two-stage strategy of SVs + SS.
| Voice Type | Model |  Accuracy | Sensitivity | Specificity| F1-Score|
| :---------: | :----: | :--------: | :----------: | :---------: | :------: |
|   SVs+SS   |AdaBoost+CNN-Transformer|  100%  |    -        | -          | -       |
