# CS598DL4H-final-project
Final Project for CS598DL4H at UIUC

## Installation
We built this model using Python version 3.9.7.

First clone this repo to your machine.

```bash
git clone https://github.com/MohiTheFish/CS598DL4H-final-project.git CS598
cd CS598
```

Next, We recommend using a virtual environment to avoid collisions with the global python installation. 
Run the following:
```bash
python -m venv venv
```

This will create a directory for your virtual environment called venv.
On Windows cmd, you will run this command to activate the venv:
```cmd
.\venv\Scripts\activate
```
**Untested*
On Unix, you will run this command to activate the venv:
```bash
source ./venv/bin/activate
```

You can deactivate the venv at any time by running the following in the command line.
```
deactivate
```

With your venv active, you can install all the necessary packages with  the following command
```
pip install -r requirements.txt
```

## Processing the Data
Before you train the model, you'll need to acquire data. We used the MIMIC-III dataset for our training, but theoretically you should be able to use any dataset that consists of EHRs that includes patient medical codes and prescription codes. 

To replicate our results, you'll first need to acquire access to the MIMIC-III dataset. You can acquire access through PhysioNet by being credentialed: https://physionet.org/settings/credentialing/. Once you have been authorized, the dataset is available here: https://physionet.org/content/mimiciii/1.4/. Feel free to download and unpack the whole dataset, but the only files we used were

* ADMISSIONS.csv
* DIAGNOSES_ICD.csv
* PATIENTS.csv
* PRESCRIPTIONS.csv

We recommend saving the files to the following directory
```
data/mimic-iii-clinical-database-1.4
```

Once you have the data in the above directory, you can run our `process_data` script
```
cd data
python process_data.py [--mimic_dir MIMIC_DIR] [--out_dir OUTPUT_DIR]
cd ..
```
If you saved the MIMIC-III data to something other than the above directory, you must specify it with the `--mimic_dir` command. 

You'll notice that there is a new directory
```
data/processed_data
```
This has all of the preprocessed data that we used. 

## Building the Model
You can build and train the model and have it be evaluated by simply running 
```bash
python coam_pytorch.py [--save_model OUTPUT_PATH]
```
The script will default to using the files we generated before. It will run for 10 epochs with a batch size of 100. If you choose to save the model, you can evaluate it later by using the load model arg:
```bash
python coam_pytorch.py [--load_model MODEL_PATH]
```
If this arg is specified, there will be no training done, and the existing model wil *only* be evaluated.

https://github.com/easyfan327/Pytorch-RETAIN