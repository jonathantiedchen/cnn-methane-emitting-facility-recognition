# Multi Class Satellite Image Classification Using CNNs For Methane Emitting Facility Tracking

Machine learning approaches can be used to map methane emissions to their sources and thus the contribution to global warming as suggested by Zhu et al. (2022). Therefore, this project aims to use deep learning techniques to classify satellite images into seven categories – Concentrated Animal Feeding Operations (CAFOs), Landfills, Mines, Negative, Processing Plants, Refineries & Terminals, WW Treatment – using a dataset with over 86,000 satellite images (Zhu et al., 2022). This is done by implementing and training Convolutional Neural Networks (CNNs), as well as using pre-trained models with transfer learning and fine-tuning.

## Data 
The used dataset initially was created to support building a global database of methane emitting infrastructure called Methane Tracking Emissions Reference (METER). This dataset contributes to the tracking of emitted volumes to their sources. It contains a total of 86,599 georeferenced images from the US labeled for the presence or absence of any of the six possible methane emitting facilities. The majority class is “Negative” with 34,195 instances, while the minority, with least instances of 1,706, is “Mines”. As can be observed from Figure 2, some labels occur significantly more often than others – therefore it is considered an imbalanced dataset. To address the imbalance in the dataset, a hybrid technique consisting of two steps is applied: Undersampling majority classes and Oversampling minority classes. After the balancing the training dataset consists of 2,000 instances for each label and therefore is perfectly balanced with a total of 14,000 images.

## Model Choice
The baseline model used is the AlexNet CNN architecture, which consists of five convolutional layer, three max pooling and three fully connected layers. To add a next layer of complexity, a self- implemented VGG16 model was used.ResNet-50 was chosen as the third model, as it yields an even higher accuracy on the ImageNet Dataset than the VGG16 model. First, the ResNet model was used for transfer learning by importing the pre-trained model provided by Keras and adding a fully connected layer, which was trained on the training samples. In a second step, fine-tuning was used, by unfreezing the last convolution block and re-trained with the training sample. 

## Training

<div style="text-align: center;">
    <img src="_images/5alexnet_image_CAFOs.png)" width="500">
</div>
