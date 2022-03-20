# INTER IIT TECH MEET 10.0
# BOSCH’S AGE AND GENDER DETECTION

**Team Name: “404 Not Found”**

**Team Member**

 - Abhishek Biswas (abhishekbiswas772@gmail.com)
 - Jayesh Lohar (jayeshpanchal063@gmail.com)
 - Aman Tejaswi (atejaswi4@gmail.com)
 - Saidutta Mohanty (sd.moh.800@gmail.com)

**SAGAN MODEL (Super Resolution GAN)**

SRGAN was proposed by Twitter researchers. The goal of this design is to recover finer textures from images when they are upscaled, so that the image's quality is not damaged. Other methods for performing this work, such as bilinear interpolation, can be implemented, but they suffer from picture information loss and smoothing. The authors presented two architectures in this paper: one without GAN (SRResNet) and one with GAN (SRGAN). It is determined that SRGAN has higher accuracy and produces more visually appealing images than SRGAN.

**Our approach to solving this problem:**

First we took some video footage, and then we wrote a Python code to convert the video into an image, and then we had to store the image in the source folder. After that, we used a pretrained SAGAN model for taring of our data. This process is called "transfer learning."
Before training the pretrained model, we had to do some basic things. Firstly, we had to import the necessary libraries into Python, which included Sys, Keras, CV2, Numpy, Skimage, and Matplotlib. And also print the version of the library that we are using in the making of the project.

After importing the library, I found some functions which calculate the image quality matrix such as peak signal to noise ratio (PSNR), mean squared error (MSE), and structural similarity (SSIM). which is required for the evaluation of the image. The structural similarity index is directly imported from the scikit-image library, but the PSNR and MSE are not present in this library, so we defined the function for the calculation of PSNR and MSE of an image. After that, we defined a function that combines the three image quality matrices, which will show the clear comparison of the image. The formula for calculating the PSNR and MSE is done by using the following expression:

After defining the function for the calculation of the image matrices, we started the preparation of images, which is required for our pretrained SAGAN model. We had made some video footage and then we wrote a code that converts the video into a series of images and stores them in some temporary folder. After that, we produced some low-quality versions of the same images. This was accomplished by resizing the image, both downwards and upwards, using OpenCV. There are several interpolation methods that can be used to resize the images. But in this project, we used bilinear interpolation. And after producing the low quality image, we had to save the images in a separate folder.

After degrading the original images, the image quality matrices are calculated and the images are correctly degraded. The calculation of PSNR, MSE, and SSIM is calculated between the reference and the degraded images that we just prepared. After that, we created the super resolution model. The super resolution model is a deep convolutional neural network that learns end-to-end mapping of low-resolution to high-resolution images. As a result, we can use it to improve the image quality of low-resolution images. Firstly, In keras, it is simple by adding the layer one after another. I also defined the optimizer using Adam and also created a function that will compile the super resolution model. The architecture and hyper parameters of the super resolution network can be obtained from the research papers.

After defining the model, now we are part of the deployment of this model. Before doing this, first we had to define a couple of image processing functions and the preprocessing of images is done extensively before using them as they are input to the network. This processing includes the cropping and colour separations. The training of deep neural networks takes time and processing power. So we used pretrained weights for the super-resolution model. These weights are found on the following GitHub page: https://github.com/MarkPrecursor/SRCNN-keras

We can now perform single-image super resolution on all of our input images after testing the network, and the PSNR, MSE, and SSIM are calculated on the images we produced.Then we could save these images directly or create subplots to conveniently display the original, low-resolution and high-resolution images (super-resolution images) side by side.

Following that, we saved the subplot compared images as well as the super-resolution images in a separate folder, and then we defined a function that will create a video from a series of images using the openCV library and save the final video in a separate folder.
After saving the final video, Now it is the part of detecting the age and gender of the character present in the image. This is also done using a pretrained model called Age/Gender.net. This classification model is based on the Caffe framework. The Age and Gender.net are used for age and gender classification, and this model was trained using the Adience-OUI dataset. This model was created by Gil Levi and Tal Hasser. The project page for the model is http://www.openu.ac.il/home/hassner/projects/cnn_agegender/.

Caffe is a deep learning framework made with expression, speed, and modularity in mind. It was developed by Berkeley AI Research (BAIR) and by community contributors. Yangqing Jia created the project during his PhD at UC Berkeley. Caffe is released under the BSD 2-Clause license. The reason behind the cafe model is due to its expressive architecture, extensible code, speed, and having a large community base.

So, for predicting and classifying gender and age, we first imported some necessary libraries, such as openCV, math, and time. After that, we have defined the age-gender coffee shop model. The network uses 3 convolutional layers, 2 fully connected layers, and a final output layer. The details of the layers are given below.

  - Conv1: The first convolutional layer has 96 nodes of kernel size 7.
  - Conv2: The second convolution layer has 256 nodes with a kernel size of 5.
  - Conv3: The third convolution layer has 384 nodes with a kernel size of 3.
  - The two fully connected layers have 512 nodes each.

For the gender prediction, the output layer in the gender prediction network is of type softmax with 2 nodes indicating the two classes "male" and "female". However, in the case of the age prediction, the problem is approached as a regression problem since we are expecting a real number as the output. However, estimating age accurately using regression is challenging. So, for age detection, eight classes are divided into the following age groups [(0 – 2), (4–6), (8–12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100)]. Thus, the age prediction network has eight nodes in the final softmax layer, indicating the mentioned age ranges.
Firstly, we created some DNN face detector functions which are used to detect faces. The model is only 2.7MB and is pretty fast, even on the CPU. After writing the face-detecting function, we have written a function that will detect and classify the gender of the images. The forward pass gives the probabilities or confidence of the two classes. We take the maximum of the two outputs and use it as the final gender prediction. After that, we load the age network and use the forward pass to get the output. Since the network architecture is similar to that of the Gender Network, we can take the maximum out of all the outputs to get the predicted age group. Then we will display the output of the final video footage, which will show the age and gender of the character in the video footage.

