# This project evaluates the performance of transfer learning using keras (with tensorflow backend) using data from the DDSM mammography dataset: 
	
#	https://www.kaggle.com/skooch/ddsm-mammography

# This uses class DeepImageBuilder to load and prepare the data in the right format, imports and builds a keras CNN architecture, and trains
# N number of models based on the imported keras CNN.  Each of the models uses the same training data, but
# ImageDataGenerator applies arbitrary transformations to the source training images. This is to evaluate how well the
# model can generalize (i.e., evaluate its robustness to transformations in image data). 

# Set the parameters of the MAIN file with configuration file (config)

# FUTURE: 
#	- change config file to json type 
#	- evaluate performance based on hyperparameter tuning
#	- evaluate performance based on systematic transformations to images
