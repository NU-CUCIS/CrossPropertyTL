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

1. Prepare a `.csv` file which contain two columns. The first column contains the compound, and the second column contains the value of target property as shown below:
 
| pretty_comp | target property |
| ----------- | --------------- |
| KH2N        | -0.40           |
| NiTeO4      | -0.82           |

2. Use the Jupyter Notebook in [`representation`](./representation) folder to pre-process and convert the first column of the `.csv` file into elemental fraction, physical attributes or to extract features using a pre-trained ElemNet model. Above example when converted into elemental fraction the becomes: 

| pretty_comp |  H  | ... | Pu | target property |
| ----------- | --- | --- | -- | --------------- |
| KH2N        | 0.5 | ... | 0  | -0.40           |
| NiTeO4      | 0   | ... | 0  | -0.82           |

3. Split the customized dataset into train, validation and test set.

We have provided an example of customized datasets in the repository: `data/sample`. Here we have converted the first column of the `.csv` file into elemental fractions. Note that this is required for both training and predicting. 

### Run ElemNet model

The code to run the ElemNet model is provided in the [`elemnet`](./elemnet) folder. In order to run the model you can pass a sample config file to the dl_regressors_tf2.py from inside of your elemnet directory:

`python dl_regressors_tf2.py --config_file sample/sample-run_example_tf2.config`

The config file defines all the related hyperparameters associated with the model training and model testing such as loss_type, training_data_path, val_data_path, test_data_path, label, input_type, etc.
For transfer learning, you need to set 'model_path' [e.g. `model/sample_model`]. 
The output log from the model training will be shown in the [`log`] folder as `log/sample-run_example_tf2.log` file. 
The trained model will be saved in [`model`](./model) folder. 
If you want to randomly initialize the weight of the last layer of the trained model as done in the work, set the `last_layer_with_weight` hyperparameter of the config file as `false`. 
You can use your own customized dataset as long as you correctly specify the training_data_path, val_data_path and test_data_path inside the config file.

The above command runs a default task with an early stopping of 200 epochs on small dataset of formation energy. This sample task can be used without any changes to give an idea of how the ElemNet model can be used. The sample task should take about 1-2 minute on an average when a GPU is available and give a test set MAE of 0.28-0.32 eV after the model training is finished.

Note: Your <a href="https://machinelearningmastery.com/different-results-each-time-in-machine-learning/">results may vary</a> given the stochastic nature of the algorithm or evaluation procedure, the size of the dataset, the target property to perform the regression modelling for or differences in numerical precision. Consider running the example a few times and compare the average outcome.

## Developer Team

The code was developed by Vishu Gupta from the <a href="http://cucis.ece.northwestern.edu/">CUCIS</a> group at the Electrical and Computer Engineering Department at Northwestern University.

## Publications

Please cite the following works if you are using ElemNet model:

1. Vishu Gupta, Kamal Choudhary, Francesca Tavazza, Carelyn Campbell, Wei-keng Liao, Alok Choudhary, and Ankit Agrawal, “Cross-property deep transfer learning framework for enhanced predictive analytics on small materials data”.

## Acknowledgements

The open-source implementation of ElemNet <a href="https://github.com/NU-CUCIS/ElemNet">here</a> have provided significant initial inspiration for the structure of this code-base.

## Disclaimer

The research code shared in this repository is shared without any support or guarantee on its quality. However, please do raise an issue if you spot something wrong or that could be improved and I will try my best to solve it.

email: vishugupts2020@u.northwestern.edu
Copyright (C) 2021, Northwestern University.
See COPYRIGHT notice in top-level directory.

## Funding Support

This work was performed under the following financial assistance award 70NANB19H005 from U.S. Department of Commerce, National Institute of Standards and Technology as part of the Center for Hierarchical Materials Design (CHiMaD). Partial support is also acknowledged from DOE awards DE-SC0014330, DE-SC0019358 and DE-SC0021399.
