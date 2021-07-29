# auto_training

This doesn't use the teachable machine but instead uses detecto which is the same library used for the crucible detect model. <br />
It is significantly slower to train but it is all automated and local so internet access is not required. <br />
Due to the lengthy training process, there is no finished model that could've been tested and so performance of this approach is completely unknown. <br />

### how to use 
* Sorted images must be in their appropriate folders (IMPORTANT: name of folders MUST BE integers and consecutively named due to the way the functions are called). <br />
* An empty folder named 'validation' is also needed. 
* 'Trainer.py' will use the current crucible detect model (which works flawlessly) and create two .csv files. 
* The .csv files are the label filed used for training and validation. It contains the image coordinates of the crucible in each photo and its powder volume (header: 'class'). 
* One .csv file is used for training and the other .csv file is used for validation. 
* The code will automatically select 10% of the images to be used for validation. Then the code will start training a new model and save it as 'new_model.pth'. 

##### notes
* Training the model with 25 images (22 used for training and 3 used for validation) took 38 minutes on a laptop without GPU acceleration. Using a dedicated graphics card will 
drastically improve the training time but it had issues with running out of VRAM that couldn't be resolved. Running the code in Google collabs with their provided GPU runtime 
worked fine and it was approximately 10 times faster. However, a human operator must be there to save the 'new_model.pth' before the runtime disconnects. Google collabs runtime will timeout 
if it's idle (no code running) for 30 minutes or if there is no user input (even if code is running) for 2 hours. <br />
* The 'path=' technically shouldn't be needed because it will be the same as the working directory but it's there for redundancy and if the trainer.py will be executed in a different
working directory.  <br />
* 'example.py' shows how to use the newly trained model. 
* Every image in the folders will create a copy into the root directory for training. This may take up a lot of storage depending on the resolution of images. 
