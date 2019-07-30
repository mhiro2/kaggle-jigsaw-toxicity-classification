import gc
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier, Pool


COL_TEXT = 'comment_text'
COL_TARGET = 'target'
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]

SEED = 2019


class JigsawEvaluator():
    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25):
        self.y = y_true
        self.y_i = y_identity
        self.n_subgroups = self.y_i.shape[1]
        self.power = power
        self.overall_model_weight = overall_model_weight

    @staticmethod
    def _compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _compute_subgroup_auc(self, i, y_pred):
        mask = self.y_i[:, i] == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bpsn_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bnsp_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y != 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def compute_bias_metrics_for_model(self, y_pred):
        records = np.zeros((3, self.n_subgroups))
        for i in range(self.n_subgroups):
            records[0, i] = self._compute_subgroup_auc(i, y_pred)
            records[1, i] = self._compute_bpsn_auc(i, y_pred)
            records[2, i] = self._compute_bnsp_auc(i, y_pred)
        return records

    def _calculate_overall_auc(self, y_pred):
        return roc_auc_score(self.y, y_pred)

    def _power_mean(self, array):
        total = sum(np.power(array, self.power))
        return np.power(total / len(array), 1 / self.power)

    def get_final_metric(self, y_pred):
        bias_metrics = self.compute_bias_metrics_for_model(y_pred)
        bias_score = np.average([
            self._power_mean(bias_metrics[0]),
            self._power_mean(bias_metrics[1]),
            self._power_mean(bias_metrics[2])
        ])
        overall_score = self.overall_model_weight * self._calculate_overall_auc(y_pred)
        bias_score = (1 - self.overall_model_weight) * bias_score
        return overall_score + bias_score

    def get_all_score(self, y_pred):
        bias_metrics = self.compute_bias_metrics_for_model(y_pred)
        power_means = [
            self._power_mean(bias_metrics[0]),
            self._power_mean(bias_metrics[1]),
            self._power_mean(bias_metrics[2])
        ]
        bias_score    = np.average(power_means)
        overall_auc   = self._calculate_overall_auc(y_pred)
        overall_score = self.overall_model_weight * overall_auc
        bias_score    = (1 - self.overall_model_weight) * bias_score
        return {
            'overall_auc': overall_auc,
            'subgroup_auc': power_means[0],
            'bpsn_auc': power_means[1],
            'bnsp_auc': power_means[2],
            'final_metrics': overall_score + bias_score,
        }


train_df = pd.read_csv(f'../input/train.csv')
test_df  = pd.read_csv(f'../input/test.csv')

# subgroup negative weighting
subgroup_bool_train = train_df[IDENTITY_COLUMNS].fillna(0) >= 0.5
toxic_bool_train = train_df['target'].fillna(0) >= 0.5
subgroup_negative_mask = subgroup_bool_train.values.sum(axis=1).astype(bool) & ~toxic_bool_train

y = (train_df[COL_TARGET].values >= 0.5).astype(np.int)
y_identity = (train_df[IDENTITY_COLUMNS].values >= 0.5).astype(np.int)

weights = np.ones((len(train_df), ))
ratio = 2.0
weights += subgroup_negative_mask * ratio


train_bert = joblib.load('./bert_base_train_embeddings.pkl')
test_bert = joblib.load('./bert_base_test_embeddings.pkl')

n_splits = 5
kfold = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(y, y))

score_list     = []
test_pred_list = []
final_list     = []
train_pred = np.zeros(len(y))

for fold, (train_idx, valid_idx) in enumerate(kfold):
    x_train = train_bert[train_idx]
    y_train = y[train_idx]
    x_valid = train_bert[valid_idx]
    y_valid = y[valid_idx]
    w_train = weights[train_idx]
    w_valid = weights[valid_idx]

    train_pool = Pool(x_train, label=y_train, weight=w_train)
    valid_pool = Pool(x_valid, label=y_valid, weight=w_valid)

    params = {
        'eval_metric': 'AUC',
        'boosting_type': 'Ordered',
        'od_type': 'Iter',
        'od_wait': 50,
        'bagging_temperature': 0.3,
        'learning_rate': 0.01,
        'iterations': 10000,
        'depth': 6,
        'l2_leaf_reg': 5.0,
        'random_seed': SEED,
        'use_best_model': True,
        'task_type': 'GPU',
    }
    train_params = {
        'early_stopping_rounds': 100,
        'verbose_eval': 100,
    }

    clf = CatBoostClassifier(**params)
    clf.fit(train_pool, eval_set=valid_pool, **train_params)

    oof_pred = clf.predict(valid_pool, prediction_type='Probability')
    oof_pred = oof_pred[:, 1]

    test_pred = clf.predict(test_bert, prediction_type='Probability')
    test_pred = test_pred[:, 1]

    score = roc_auc_score(y_valid, oof_pred)
    print(score)

    score_list.append(score)
    test_pred_list.append(test_pred)
    train_pred[valid_idx] = oof_pred

    evaluator = JigsawEvaluator(y_valid, y_identity[valid_idx, :])
    auc_score = evaluator.get_all_score(oof_pred)
    print(pd.Series(auc_score))
    final_score = auc_score['final_metrics']
    final_list.append(final_score)

    clf.save_model(f'cb_base_fold{fold+1}_v{SEED}.txt')

final_score = np.mean(final_list)
print(final_score)


test_id_list = test_df['id'].values
prediction = np.mean(test_pred_list, axis=0)
submission = pd.DataFrame.from_dict({
   'id': test_id_list,
   'prediction': prediction
})
submission.to_csv(f'submission.csv', index=False)

