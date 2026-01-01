# Obstacle_Detection_using_LiDAR_and_Machine_Learning
This includes the code used to train the CNN Model for predicting potential threats for railway operations using image frames captured by a LiDAR sensor

model.py - has the main CNN architecture with 4 convolutional blocks with batch normalization and fully connected layers for classification
data_loader.py - splits the data into train/test/validation (70/15/15 by default), applies augmentation and resizes the image to 224x224
train.py - this is the main training script with Validation monitoring, Early stopping, Learning rate scheduling, Model checkpointing and test set evaluation
inference.py - used to make predictions on unseen data either for a single frame or multiple frames together
requirements.txt - Dependencies (PyTorch, torchvision, etc.)
test_model.py - Quick test script to verify model architecture
example_usage.py - Usage examples and documentation

To use the model
Install dependencies:
   pip install -r requirements.txt

Train the model:
   python train.py
Or with custom parameters:
   python train.py --batch-size 64 --epochs 100 --learning-rate 0.0001

Use the below code to test the model

To run on multiple frames together
python inference.py --model-path checkpoints/best_model.pth --image-dir Data/threat/ --output results.json

To run on one frame
python inference.py --model-path checkpoints/best_model.pth --image-path Data/threat/ss1.png

Find the link below for the dataset used
https://drive.google.com/drive/folders/1NImh-GdXEseTdd9qUjNx1_S1O6GrALoP?usp=drive_link
