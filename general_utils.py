import os
import pickle
import gzip
import copy
import random, time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from datetime import datetime, date
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix


    
def flatten(t):
    return [item for sublist in t for item in sublist]


def format_IR(IR_num, split_type):
    """
    make a string, which can be used when storing trained neural network, results, log files, etc.
    """
    if isinstance(IR_num[0], list):  # Check if the first element is a list
        inner = ''.join(str(num) for num in IR_num[0])
        outer = '{:02d}'.format(IR_num[1])
        return f'T{split_type}_IR_{inner}_{outer}'
    else:
        return f'T{split_type}_IR_' + '_'.join('{:03d}'.format(num) for num in IR_num)

    

def make_folders(year_start, split_type, num_class, addition_str):
    """
        create folders and subfolders
        year_start is the train start year, e.g., 2016 for predicting 2019, the year_start is 2016
        split_type is used for setting whether train conditionally or not
        note: num_class is always setting to 2, due to binary classfication
        As an example: year_start=2016, split_type=0; num_class=2; addition_str='train':
        folder: 2016_train, 
        subfolders: t0_c2_log, t0_c2_net, t0_c2_loss, t0_c2_curve, t0_c2_result   
    """
    parent_folder = str(year_start)+"_"+ addition_str
    if not os.path.exists(parent_folder):
        os.mkdir(parent_folder)
        
    log_folder = os.path.join(parent_folder, f"t{split_type}_c{num_class}_log")
    net_folder = os.path.join(parent_folder, f"t{split_type}_c{num_class}_net")
    train_folder = os.path.join(parent_folder, f"t{split_type}_c{num_class}_loss")
    figure_folder = os.path.join(parent_folder, f"t{split_type}_c{num_class}_curve")
    result_folder = os.path.join(parent_folder, f"t{split_type}_c{num_class}_result")

    try:
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        if not os.path.exists(net_folder):
            os.mkdir(net_folder)

        if not os.path.exists(train_folder):
            os.mkdir(train_folder)

        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

    except FileExistsError:
        pass

    save_folders = [net_folder, train_folder, figure_folder, result_folder]
    return save_folders, log_folder


######### Plots ###############
def calculate_plot_ROC(true_labels, nn_outputs, user_parameter, figure_name, save_figure_folder):
    """
    Plot the ROC curve for binary classification.

    Parameters:
    - true_labels: Ground truth binary labels.
    - nn_outputs: Raw outputs from the neural network.
    - user_parameter: some user parameters whcih are num_class, IR_num, split_type, out_norm; not used here, can be removed
    - figure_name: stored figure name
    - save_figure_folder: the folder to store the figure, which is usually defined from t0_c2_curve created from make_folders()

    return:
    auc_score_number: the AUC value
    """
    num_class, IR_num, split_type, out_norm = user_parameter
    figure_path=os.path.join(save_figure_folder, figure_name)
    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(true_labels, nn_outputs)
    roc_auc = auc(fpr, tpr)

    auc_score_number = roc_auc_score(true_labels, nn_outputs)
    
    # Plot the ROC curve
    plt.figure()
    lw = 1.5  # Line width
    plt.plot(fpr, tpr, color='blue', lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label='baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(figure_path,dpi=600)
    plt.show()
    plt.close()
    
    return auc_score_number
