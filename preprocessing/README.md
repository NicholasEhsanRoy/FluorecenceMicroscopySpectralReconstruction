# Preprocessing Workflow

### Note:
The example 

## Purpose
This directory is intended to contain the scripts related to preprocessing the data that trains the Pix2Pix and UNET models.

## Virtual Environment
It is reccomended that you run this all in a virtual environment. If on linux, you can create one and install the right packages using the commands:

```console
foo@bar:~path/to/FluorecenceMicroscopySpectralReconstruction$ python3 -m venv fluoro-venv
foo@bar:~path/to/FluorecenceMicroscopySpectralReconstruction$ source fluoro-venv/bin/activate
(fluoro-venv) foo@bar:~path/to/FluorecenceMicroscopySpectralReconstruction$ pip install -r requirements.txt
```

**NOTE 1: Make sure you are in the root directory of this project**

**NOTE 2: If you don't have an NVIDIA GPU on your system, use the preprocessing-requirements.txt file instead. You won't be able to train models on your system directly (but will still be able to preprocess your data locally), we suggest you setup something with a cloud service like paperspace or AWS Sagemaker. If you have a GPU but it is low on VRAM (16GB)you will likely have to do this too (but can still just use requirements.txt)**

## Workflow

1. **aviToPng.py**: If you have AVI images, you can turn them into png images by running the **aviToPng.py** script. In the top of the file, you must set 3 variables. (a) **avi_folder**. This is a root folder, whose subfolders should be avi files. (b) **file_names** These are really the folders that contain your avi files. (c) **frames_directory_base** This is the root directory where you want to store the png files. The script will create the folders in **file_names** and put the png images of the associated avi file in there.

2. **threshold-tune.py**: This script is intended to remove crosstalk. You must set the **img_file** variable. You can find it near the top of the file (just under the **on_trackbar()** method). The script will open the image you specify with the **img_file** variable and give you a slider to test different threshold values in real time. Find a threshold that keeps as much of what you're trying to catch in that colour channel as possible while minimizing the noise. You will use this threshold in **step (3)**

3. **preprocess.py**: This is the main data augmentation script. This will augment the data to increase the dataset's diversity and thereby increase the robustness of the ultimately trained models. It does this by taking each **frames** folder (with the png images), looping through them, removing crosstalk from the specified folder (using the threshold from step (2)), splitting them into 5x5 evenly sized tiles, and shuffling associated frames equally. You must set the variables: (a)**crosstalk_threshold** (what you get from step (2)), **frames_dir** (the root directory, potentially from step (1), that contains the folders with each of your channels), **output_dir** (The root directory where you want the augmented data to go), **CS_IDX** (the index of the frames folder you want to have the crosstalk removed from when the directories of frames_dir are numero-alphabetically ordered), and finally **MA_IDX** (the index of the frames folder you want to be the input in your model (so crosstalk still in it) when the directoreis of frames_dir are numero-alphabetically ordered).

4. **train_test_split.py**. This script will take your augmented data and randomly split them into a training and testing set, ensuring that frames match based on their names. You must set the variables: (a) **NUM_FILES** (the number of total frames in the augmented data folder you want to include), (b) **TEST_PERC** (the percentage of the files in your dataset that you want to be reserved for testing. **NOTE: The Machine Learning standard is 20%, so it is reccomended you keep that the same.**), and finally **source_dir** (the root directory where you have the directories containing your augmented frames from step (3))