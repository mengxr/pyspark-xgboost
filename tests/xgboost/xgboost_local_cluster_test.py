import random
import uuid

import numpy as np
from pyspark.ml.linalg import Vectors

from sparkdl.xgboost import XgboostRegressor
from tests.tests import SparkDLLocalClusterTestCase
import logging
logging.getLogger("py4j").setLevel(logging.INFO)


class XgboostClusterTest(SparkDLLocalClusterTestCase):

    def _get_max_num_concurrent_tasks(self, sc):
        """Gets the current max number of concurrent tasks."""
        # spark 3.1 and above has a different API for fetching max concurrent tasks
        if sc._jsc.sc().version() >= '3.1':
            return sc._jsc.sc().maxNumConcurrentTasks(
                sc._jsc.sc().resourceProfileManager().resourceProfileFromId(0)
            )
        return sc._jsc.sc().maxNumConcurrentTasks()

    def setUp(self):
        random.seed(2020)
        self.session.sparkContext.parallelize(list(range(4)), 4).collect()
        self.n_workers = self._get_max_num_concurrent_tasks(self.session.sparkContext)
        # The following code use xgboost python library to train xgb model and predict.
        #
        # >>> import numpy as np
        # >>> import xgboost
        # >>> X = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]])
        # >>> y = np.array([0, 1])
        # >>> reg1 = xgboost.XGBRegressor()
        # >>> reg1.fit(X, y)
        # >>> reg1.predict(X)
        # array([8.8363886e-04, 9.9911636e-01], dtype=float32)
        # >>> def custom_lr(boosting_round, num_boost_round):
        # ...     return 1.0 / (boosting_round + 1)
        # ...
        # >>> reg1.fit(X, y, callbacks=[xgboost.callback.reset_learning_rate(custom_lr)])
        # >>> reg1.predict(X)
        # array([0.02406833, 0.97593164], dtype=float32)
        # >>> reg2 = xgboost.XGBRegressor(max_depth=5, n_estimators=10)
        # >>> reg2.fit(X, y)
        # >>> reg2.predict(X, ntree_limit=5)
        # array([0.22185263, 0.77814734], dtype=float32)
        self.reg_params = {'max_depth': 5, 'n_estimators': 10, 'ntree_limit': 5}
        self.reg_df_train = self.session.createDataFrame([
            (Vectors.dense(1.0, 2.0, 3.0), 0),
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1)
        ], ["features", "label"])
        self.reg_df_test = self.session.createDataFrame([
            (Vectors.dense(1.0, 2.0, 3.0), 0.0, 0.2219, 0.02406),
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1.0, 0.7781, 0.9759)
        ], ["features", "expected_prediction", "expected_prediction_with_params",
            "expected_prediction_with_callbacks"])

    def test_regressor_basic_with_params(self):
        regressor = XgboostRegressor(**self.reg_params)
        model = regressor.fit(self.reg_df_train)
        pred_result = model.transform(self.reg_df_test).collect()
        for row in pred_result:
            self.assertTrue(
                np.isclose(row.prediction,
                           row.expected_prediction_with_params, atol=1e-3)
            )

    def get_local_tmp_dir(self):
        return "/tmp/xgboost_local_test/" + str(uuid.uuid4())

    def test_callbacks(self):
        from xgboost.callback import LearningRateScheduler
        path = self.get_local_tmp_dir()

        def custom_learning_rate(boosting_round):
            return 1.0 / (boosting_round + 1)

        cb = [LearningRateScheduler(custom_learning_rate)]
        regressor = XgboostRegressor(callbacks=cb)

        # Test the save/load of the estimator instead of the model, since
        # the callbacks param only exists in the estimator but not in the model
        regressor.save(path)
        regressor = XgboostRegressor.load(path)

        model = regressor.fit(self.reg_df_train)
        pred_result = model.transform(self.reg_df_test).collect()
        for row in pred_result:
            self.assertTrue(
                np.isclose(row.prediction,
                           row.expected_prediction_with_callbacks, atol=1e-3)
            )
