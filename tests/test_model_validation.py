import unittest
import numpy as np
import pandas as pd
from model_validation import (FEATURES, chronological_split, expanding_date_folds,
                              load_data, preprocessing, score_predictions)


class ValidationTests(unittest.TestCase):
    def test_dates_and_ids_do_not_cross_holdout(self):
        train, test = chronological_split(load_data())
        self.assertLess(train.Date.max(), test.Date.min())
        self.assertFalse(set(train.FlightNumber) & set(test.FlightNumber))

    def test_same_date_stays_together(self):
        frame = load_data()
        frame.loc[72, 'Date'] = frame.loc[71, 'Date']
        train, test = chronological_split(frame)
        self.assertFalse(set(train.Date) & set(test.Date))

    def test_temporal_tuning_excludes_holdout(self):
        train, test = chronological_split(load_data())
        train = train.reset_index(drop=True)
        for a, b in expanding_date_folds(train):
            self.assertLess(train.iloc[a].Date.max(), train.iloc[b].Date.min())
            self.assertLess(train.iloc[b].Date.max(), test.Date.min())

    def test_scaler_never_fits_test_outlier(self):
        frame = load_data()
        train, test = chronological_split(frame)
        transform = preprocessing().fit(train[FEATURES])
        scaler = transform.named_transformers_['numeric'].named_steps['scaler']
        expected = train.FlightNumber.mean()
        self.assertAlmostEqual(scaler.mean_[0], expected)
        test.loc[:, 'FlightNumber'] = 1000000
        transform.transform(test[FEATURES])
        self.assertAlmostEqual(scaler.mean_[0], expected)

    def test_baseline_metrics_expose_class_imbalance(self):
        actual = np.array([0, 0, 1, 1, 1])
        result = score_predictions(actual, np.ones(5))
        self.assertEqual(result['accuracy'], .6)
        self.assertEqual(result['balanced_accuracy'], .5)
        self.assertEqual((result['tn'], result['fp'], result['fn'], result['tp']), (0, 2, 0, 3))

    def test_leaking_and_unverified_columns_excluded(self):
        self.assertFalse(set(FEATURES) & {'Outcome', 'Class', 'PayloadMass', 'LandingPad', 'Serial', 'ReusedCount'})


if __name__ == '__main__':
    unittest.main()
