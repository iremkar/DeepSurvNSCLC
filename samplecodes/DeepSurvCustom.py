# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Sim-Irem
#

# +
# Use DeepSurv from the repo

import sys
sys.path.append("/irem/code/DeepSurv/veriler/veriler/DeepSurv/deepsurv")
import deep_surv

from deepsurv_logger import DeepSurvLogger, TensorboardLogger
import utils
import viz

import numpy as np
import pandas as pd

import lasagne
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
# -
n_epochs = 2000;
L2_reg=0.01;
batch_norm=True;
dropout=0.2;
hidden_layers_sizes=[100,10];
learning_rate=0.0001;
lr_decay=0.0001;
momentum=0.9;
standardize=True
#csv= '/mnt/batch/tasks/shared/LS_root/mounts/clusters/survival3/code/DeepSurv/i.csv'
logdir = '/irem/code/DeepSurv/veriler/veriler/DeepSurv/logs2/'


i=0;
for i in range(100):
    i<=i+1

    import glob 
    files = glob.glob("/irem/code/DeepSurv/veriler/sim21/train*.csv")

    import glob 
    valid = glob.glob("/irem/code/DeepSurv/veriler/sim21/valid*.csv")

    train_dataset_fp = files[i]
    train_df = pd.read_csv(train_dataset_fp)
    train_df.head()

    valid_dataset_fp = valid[i]
    valid_df = pd.read_csv(valid_dataset_fp)
    valid_df.head()


    j=0;
    for j in range(1):
        j=j+1
        # event_col is the header in the df that represents the 'Event / Status' indicator
        # time_col is the header in the df that represents the event time
        def dataframe_to_deepsurv_ds(df, event_col = 'Event', time_col = 'Time'):
            # Extract the event and time columns as numpy arrays
            e = df[event_col].values.astype(np.int32)
            t = df[time_col].values.astype(np.float32)

            # Extract the patient's covariates as a numpy array
            x_df = df.drop([event_col, time_col], axis = 1)
            x = x_df.values.astype(np.float32)

            # Return the deep surv dataframe
            return {
                'x' : x,
                'e' : e,
                't' : t
            }

        # If the headers of the csv change, you can replace the values of 
        # 'event_col' and 'time_col' with the names of the new headers
        # You can also use this function on your training dataset, validation dataset, and testing dataset
        train_data = dataframe_to_deepsurv_ds(train_df, event_col = 'Event', time_col= 'Time');

        valid_data = dataframe_to_deepsurv_ds(valid_df, event_col = 'Event', time_col= 'Time');

        hyperparams = {
            'L2_reg': L2_reg,
            'batch_norm':batch_norm,
            'dropout': dropout,
            'hidden_layers_sizes': hidden_layers_sizes,
            'learning_rate': learning_rate,
            'lr_decay': lr_decay,
            'momentum': momentum,
            'n_in': train_data['x'].shape[1],
            'standardize': standardize
        }

        # Create an instance of DeepSurv using the hyperparams defined above

        model = deep_surv.DeepSurv(**hyperparams)

        # DeepSurv can now leverage TensorBoard to monitor training and validation
        # This section of code is optional. If you don't want to use the tensorboard logger
        # Uncomment the below line, and comment out the other three lines: 
        # logger = None

        experiment_name = 'test_experiment_erdal'
        logdir = logdir
        logger = TensorboardLogger(experiment_name, logdir=logdir)

        # Now we train the model
        update_fn=lasagne.updates.nesterov_momentum # The type of optimizer to use. \
                                                    # Check out http://lasagne.readthedocs.io/en/latest/modules/updates.html \
                                                    # for other optimizers to use
        n_epochs = n_epochs

        # If you have validation data, you can add it as the second parameter to the function
        metrics = model.train(train_data,valid_data,n_epochs=n_epochs, logger=logger, update_fn=update_fn)

        # Print the final metrics
        print(i,j,'Train C-Index:', metrics['c-index'][-1])
        print(i,j,'Valid C-Index: ',metrics['valid_c-index'][-1])
        print(i,j,'Train loss: ',metrics['loss'][-1])
        print(i,j,'Validation loss: ',metrics['valid_loss'][-1])
        print(i,j)

        # Plot the training / validation curves
        #viz.plot_log(metrics)

