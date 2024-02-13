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
    
    if isinstance(IR_num[0], list):  # Check if the first element is a list
        inner = ''.join(str(num) for num in IR_num[0])
        outer = '{:02d}'.format(IR_num[1])
        return f'T{split_type}_IR_{inner}_{outer}'
    else:
        return f'T{split_type}_IR_' + '_'.join('{:03d}'.format(num) for num in IR_num)

    

def make_folders(year_start, split_type, num_class, addition_str):
    
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
    - nn_outputs: Raw outputs (logits) from the neural network.
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
    
    
    
def calculate_ROC_multi_class(outputs, class_labels, user_parameter, figure_name, save_figure_folder):
    
    num_class, IR_num, split_type, out_norm = user_parameter
    figure_path = os.path.join(save_figure_folder, figure_name+'.png')
    labels = class_labels

    # Compute ROC curve and ROC area for each class
    n_classes = outputs.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), outputs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    all_vals = []
    plt.figure()
    
    if num_class <= 2:
        plt.plot(fpr[1], tpr[1], label=f'AUC_ROC={roc_auc[1]}')
        average_auc = roc_auc[1]
        
        for i in range(n_classes):
            all_vals.append(roc_auc[i])
    else:
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'class-{i}: AUC_ROC={roc_auc[i]}')
            all_vals.append(roc_auc[i])
        average_auc = np.mean(all_vals)
        
        # Plot macro-average ROC curve
    plt.plot(fpr["macro"], tpr["macro"], label=f'macro-average: AUC_ROC={roc_auc["macro"]}', color='navy', linestyle=':')
    plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average: AUC_ROC={roc_auc["micro"]}', color='deeppink', linestyle='--')

    
    plt.plot([0, 1], [0, 1], 'k--', label='baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC: AUC={average_auc}')
    plt.legend(loc="lower right")
    plt.savefig(figure_path, dpi=600)
    plt.show()
    plt.close()
    return all_vals, average_auc

        
def calculate_PR_multi_class(outputs, class_labels, user_parameter, figure_name, save_figure_folder):
    
    num_class, IR_num, split_type, out_norm = user_parameter
    figure_path=os.path.join(save_figure_folder, figure_name+'.png')
    
    labels = class_labels

    # Compute Precision-Recall curve and area for each class
    n_classes = outputs.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i], outputs[:, i])
        average_precision[i] = average_precision_score(labels[:, i], outputs[:, i])

    # Compute micro-average Precision-Recall curve and area (optional)
    precision["micro"], recall["micro"], _ = precision_recall_curve(labels.ravel(), outputs.ravel())
    average_precision["micro"] = average_precision_score(labels.ravel(), outputs.ravel())

    # Plot all Precision-Recall curves
    all_vals=[]
    plt.figure()
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label=f'class-{i}: AUC={average_precision[i]}')
        all_vals.append(average_precision[i])
        
    average_ap = np.mean(all_vals)
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR curve: AUC={average_ap}')
    plt.legend(loc="lower right")
    plt.savefig(figure_path,dpi=600)
    plt.show()
    plt.close()
    return all_vals, average_ap

  

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, figure_name="cm_plot", save_figure_folder="save_aucPlot"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    figure_path=os.path.join(save_figure_folder, figure_name+'.png')
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalized cm


    fig, ax = plt.subplots(figsize=(5, 5)) 
        
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)  # always plot cm_norm
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    fmt = '{:.2%}\n{:,}'  # format for percentage and count
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, fmt.format(cm_norm[i, j], cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    plt.savefig(figure_path,dpi=600)
    plt.show()
    plt.close()
    return cm