# -*- coding: utf-8 -*- #
# Copyright 2021
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility script for SemEval Tasks"""
"""*********************************************************************************************"""
#   Synopsis     [ Scripts for    ]
#   Author       [ Shammur A Chowdhury ]

"""*********************************************************************************************"""

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error
from scipy import stats
import statistics


## Function to create train test from the folds given
## in the task data
def get_train_dev(test_fold_id, table):
    test = table[test_fold_id]
    tables = []
    index = 0
    while index < len(table):

        if index != test_fold_id:
            tables.append(table[index])
        index += 1
    train = pd.concat(tables, ignore_index=True)
    return train, test


# Function to merge all the folds and return one unified dataframe
def get_train(table):
    train = pd.concat(table, ignore_index=True)
    return train


# Functions for post-processing json with results
def process_eval_jsons_task1(eval_jsons):
    evaluations = {}

    for info in eval_jsons:
        for ev in info.keys():
            if ev not in evaluations:
                evaluations[ev] = []
            evaluations[ev].append(info[ev])
    for ev in evaluations.keys():
        val = statistics.mean(evaluations[ev])
        std = statistics.stdev(evaluations[ev])
        evaluations[ev].append(str(round(val, 3)) + " (±" + str(round(std, 2)) + ")")

    return pd.DataFrame.from_dict(evaluations)


def process_eval_jsons_task2(eval_jsons):
    evaluations = {'mse': [], 'rmse': [], 'rho': []}
    pred = []
    label = []
    #   print("Total entries in Json dict:", len(eval_jsons))
    for info in eval_jsons:
        evaluations['mse'].append(info['mse'])
        evaluations['rmse'].append(info['rmse'])
        evaluations['rho'].append(info['rho'])
        pred.extend(info['eval_pred'])
        label.extend(info['eval_labels'])

    #   print(len(evaluations['rmse']))
    for ev in evaluations.keys():
        mean_val = round(statistics.mean(evaluations[ev]), 3)
        # evaluations[ev].append(mean_val)
        std_val = round(statistics.stdev(evaluations[ev]), 2)
        evaluations[ev].append(str(mean_val) + "(±" + str(std_val) + ")")
    rho_overall, pval = stats.spearmanr(label, pred)
    # print(rho_overall)

    return pd.DataFrame.from_dict(evaluations), rho_overall


## Evaluation function for both the task
def compute_metrics_task1(preds, labels):
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')

    return [
               'accuracy:{}'.format(acc),
               'Precision:{}'.format(pre),
               'Recall:{}'.format(recall),
               'F1:{}'.format(f1),
           ], {
               'accuracy': acc,
               'Precision': pre,
               'Recall': recall,
               'F1': f1,
           }

def compute_metrics_task2(preds, labels):
    # calculate accuracy using sklearn's function
    f1 = f1_score(labels, preds, average='micro')

    return [
               'F1:{}'.format(f1),
           ], {
               'F1': f1,
           }
def compute_metrics_task3(preds, labels):
    # calculate accuracy using sklearn's function
    f1 = f1_score(labels, preds, average='macro')

    return [
               'F1:{}'.format(f1),
           ], {
               'F1': f1,
           }
