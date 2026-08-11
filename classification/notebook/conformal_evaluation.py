# file: conformal_evaluation.py

import numpy as np


class ClassificationEvaluator:
    def evaluate(self, pred_sets, y_true):
        y_true = np.asarray(y_true)

        coverage = np.mean([
            y_true[i] in pred_sets[i]
            for i in range(len(y_true))
        ])

        avg_set_size = np.mean([
            len(pred_set)
            for pred_set in pred_sets
        ])

        return {
            "coverage": float(coverage),
            "avg_set_size": float(avg_set_size),
        }

    def evaluate_per_class(self, pred_sets, y_true):
        y_true = np.asarray(y_true)
        results = {}

        for cls in np.unique(y_true):
            idx = np.where(y_true == cls)[0]
            cls_coverage = np.mean([y_true[i] in pred_sets[i] for i in idx])
            cls_avg_set_size = np.mean([len(pred_sets[i]) for i in idx])

            results[cls] = {
                "coverage": float(cls_coverage),
                "avg_set_size": float(cls_avg_set_size),
            }

        return results