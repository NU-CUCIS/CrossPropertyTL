# ElemNet2.0

This repository contains the code for performing cross-property deep transfer learning framework to predict materials properties using elemental fraction (EF), physical attribute (PA) or extracted features as the model input. The code provides the following functions:

* Preprocess the customized dataset 
* Train a ElemNet model with a customized dataset.
* Use a pre-trained ElemNet model to perform transfer learning on customized dataset.
* Use a pre-trained ElemNet model to perform features extraction on customized dataset.
* Predict material properties of new compound with a pre-trained ElemNet model.

It is recommended to train large dataset (e.g. OQMD, MP) from scratch (SC) and small datasets (DFT-computed or experimental datasets) using transfer learning methods.

Please look at the following paper to read about the details of the cross-property deep transfer learning framework:

Cross-property deep transfer learning framework for enhanced predictive analytics on small materials data


## Installation Requirements

The basic requirement for using the files are a Python 3.6.3 Jupyter environment with the packages listed in `requirements.txt`.

## Source Files
  
Here is a brief description about the folder content:

* [`elemnet`](./elemnet): code for training ElemNet model from scratch or using a pretrained ElemNet model to perform transfer learning.

* [`data`](./data): the datasets used for training ElemNet model.

* [`representation`](./representation): Jupyter Notebook to perform feature extraction from a specific layer of pre-trained ElemNet model. We have also provided the code to convert compound into elemental fraction and physical attributes.

* [`prediction`](./prediction): Jupyter Notebook to perform prediction using the pre-trained model for ElemNet model.

## Example Use

### Create a customized dataset

To use different representations (such as elemental fractions, physical attributes or extracted features) as input to ElemNet, you will need to create a customized dataset using a `.csv` file. You can prepare a customized dataset in the following manner:

* Prepare a `.csv` file which contain two columns. The first column contains the compound, and the second column contains the value of target property. 

* Use the Jupyter Notebook in [`representation`](./representation) folder to pre-process and convert the first column of the `.csv` file into elemental fraction & physical attributes or to extract features using a pre-trained ElemNet model

* Split the customized dataset into train, validation and test set 

We have provided an example of customized datasets in the repository: `data/sample`. Here we have converted the first column of the `.csv` file into elemental fractions. Note that this is required for both training and predicting. 

### Run ElemNet model

The code to run the ElemNet model is provided in the [`elemnet`](./elemnet) folder. In order to run the model you can pass a sample config file to the dl_regressors_tf2.py from inside of your elemnet directory:

`python dl_regressors_tf2.py --config_file sample/sample-run_example_tf2.config`

The config file defines all the related hyperparameters associated with the model training and model testing such as loss_type, training_data_path, test_data_path, label, input_type [elements_tl for ElemNet] etc. For transfer learning, you need to set 'model_path' [e.g. `model/sample_model`]. The output log
from will be shown in the [`log`] folder as `log/sample.log` file. The trained model will be saved in [`data`](./model) folder.

## Developer Team

The code was developed by Vishu Gupta from the <a href="http://cucis.ece.northwestern.edu/">CUCIS</a> group at the Electrical and Computer Engineering Department at Northwestern University.

## Publications

Please cite the following works if you are using ElemNet model:

1. Vishu Gupta, Kamal Choudhary, Francesca Tavazza, Carelyn Campbell, Wei-keng Liao, Alok Choudhary, and Ankit Agrawal, “Cross-property deep transfer learning framework for enhanced predictive analytics on small materials data”.

## Questions/Comments

email: vishugupts2020@u.northwestern.edu or ankitag@eecs.northwestern.edu</br>
Copyright (C) 2021, Northwestern University.<br/>
See COPYRIGHT notice in top-level directory.

## Funding Support

This work was performed under the following financial assistance award 70NANB19H005 from U.S. Department of Commerce, National Institute of Standards and Technology as part of the Center for Hierarchical Materials Design (CHiMaD). Partial support is also acknowledged from DOE awards DE-SC0014330, DE-SC0019358 and DE-SC0021399.
