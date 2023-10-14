# SuperTuxImageSegmentation
A project for doing both image segmentation and global object classification in the 3D Game SuperTuxKart.

Both CNN and FCN mdoels are constructed in a modular blocked architecture, increasing channel dimensions while decreasing spatial dimensions of activation maps as they progress through the model. 

The FCN implementation draws inpiration from U-Net architecture, utilizing both residual and skip connections.
