# Early Stroke Diagnosis and Evaluation Based on Pathological Voice Classification Using Speech Enhancement
This paper proposes a cascaded framework for early stroke diagnosis (ESD) combining speech enhancement (based on SEWUNet) and pathological voice recognition. It processes sustained vowels (SVs) with handcrafted features and machine learning algorithms, and spontaneous speech (SS) with mel-spectrogram features and a CNN-Transformer model. Clinical trials conducted using self-built data from the paper showed that the two-stage strategy (SVs followed by SS) achieved an accuracy rate of 100%, demonstrating its potential as an auxiliary diagnostic tool.
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
│ ├── ss_train.py # Loading, preprocessing, segmentation, and enhancement of spontaneous speech data
│ ├── sv_train.ipynb # Loading, preprocessing, partitioning, and enhancement of sustained vowels data
│ ├── pytorchtools.py # Tools for stopping training early
│ ├── SerialNet.py # Two models, serial_cnn_transformer and serial_resnet_transformer, have been defined
│ ├── sparse_image_warp_pytorch.py # Sparse image deformation implementation
│ ├── spec_augment_pytorch.py # SpecAugment implementation
├── SEWUNet/
│ ├── autoencoder_SEWUNet.ipynb # Construct and train a speech enhancement model (SEWUNet) based on an autoencoder
│ ├── denoising_SEWUNet.ipynb # The main file, encompassing data loading, model definition, training, and validation processes
│ ├── parts.py # The various components of the SEWUNet model are defined
│ ├── utils.py # Includes auxiliary functions and custom classes
```
## Training
```
python ss_train.py
python sv_train.py
denoising_SEWUNet.py
```
