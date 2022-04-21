import contextlib
import logging
import shutil
import subprocess
import sys
import tempfile

import unittest

from six import StringIO

from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.taskcontext import TaskContext


@contextlib.contextmanager
def patch_stdout():
    """patch stdout and give an output"""
    sys_stdout = sys.stdout
    io_out = StringIO()
    sys.stdout = io_out
    try:
        yield io_out
    finally:
        sys.stdout = sys_stdout


@contextlib.contextmanager
def patch_logger(name):
    """patch logger and give an output"""
    io_out = StringIO()
    log = logging.getLogger(name)
    handler = logging.StreamHandler(io_out)
    log.addHandler(handler)
    try:
        yield io_out
    finally:
        log.removeHandler(handler)


class PythonUnitTestCase(unittest.TestCase):
    # We try to use unittest2 for python 2.6 or earlier
    # This class is created to avoid replicating this logic in various places.
    pass


class TestSparkContext(object):
    @classmethod
    def setup_env(cls, deploy_mode='local'):
        if deploy_mode == 'local':
            conf_path = 'dev/spark_local_conf/spark-defaults.conf'
        elif deploy_mode == 'cluster':
            conf_path = 'dev/spark_cluster_conf/spark-defaults.conf'
        elif deploy_mode == 'local-cluster':
            conf_path = 'dev/spark_local_cluster_conf/spark-defaults.conf'
        else:
            raise ValueError(f'Unknown deploy mode {deploy_mode}')

        builder = SparkSession.builder.appName('SparkDL Tests')
        with open(conf_path) as f:
            for line in f:
                l = line.strip()
                if l:
                    k, v = l.split(None, 1)
                    builder.config(k, v)
        spark = builder.getOrCreate()
        if deploy_mode == 'cluster':
            # We run a dummy job so that we block until the workers have connected to the master
            spark.sparkContext.parallelize(range(2), 2).barrier().mapPartitions(lambda _: []).collect()

        logging.getLogger('pyspark').setLevel(logging.INFO)


        cls.sc = spark.sparkContext
        cls.sql = SQLContext(cls.sc)
        cls.session = spark

    @classmethod
    def tear_down_env(cls):
        cls.session.stop()
        cls.session = None
        cls.sc.stop()
        cls.sc = None
        cls.sql = None


class TestTempDir(object):
    @classmethod
    def make_tempdir(cls, dir="/tmp"):
        """
        :param dir: Root directory in which to create the temp directory
        """
        cls.tempdir = tempfile.mkdtemp(prefix="sparkdl_tests", dir=dir)

    @classmethod
    def remove_tempdir(cls):
        shutil.rmtree(cls.tempdir)


class SparkDLTestCase(TestSparkContext, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.setup_env(deploy_mode='local')

    @classmethod
    def tearDownClass(cls):
        cls.tear_down_env()

    def assertDfHasCols(self, df, cols=[]):
        map(lambda c: self.assertIn(c, df.columns), cols)


class SparkDLLocalClusterTestCase(TestSparkContext, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.setup_env(deploy_mode='local-cluster')

    @classmethod
    def tearDownClass(cls):
        cls.tear_down_env()


class SparkDLClusterTestCase(TestSparkContext, TestTempDir, unittest.TestCase):
    """
    This test class is for tests which use SSH in a Docker-based cluster.
    It also creates a temp directory in the mounted sparkdl directory which is shared
    across the Spark driver and executors in the cluster.
    """

    @classmethod
    def setUpClass(cls):
        cls.setup_env(deploy_mode='cluster')
        cls.make_tempdir(dir="/mnt/sparkdl/dev")

        # We test on open-source spark, patch _get_cuda_visible_devices_env to
        # make sparkdl compatible with open-source spark
        def _get_cuda_visible_devices_env():
            taskContext = TaskContext.get()
            # pylint: disable=unsubscriptable-object
            addresses = [addr.strip() for addr in taskContext.resources()["gpu"].addresses]
            return ",".join(addresses)

        # We run a dummy job so that we block until the workers have connected to the master
        cls.sc.parallelize(range(4), 4).barrier().mapPartitions(lambda _: []).collect()

    @classmethod
    def tearDownClass(cls):
        cls.remove_tempdir()
        cls.tear_down_env()


class SparkDLTempDirTestCase(SparkDLTestCase, TestTempDir):

    @classmethod
    def setUpClass(cls):
        super(SparkDLTempDirTestCase, cls).setUpClass()
        cls.make_tempdir()

    @classmethod
    def tearDownClass(cls):
        super(SparkDLTempDirTestCase, cls).tearDownClass()
        cls.remove_tempdir()


class SparkDLClusterTestCaseTest(SparkDLClusterTestCase):

    def test_cluster_setup(self):
        """Test that data is physically located at different hostnames"""
        def get_hostname(_=None):
            return [subprocess.check_output(["hostname"])]

        worker_hostnames = self.sc.parallelize(range(2), 2).barrier() \
            .mapPartitions(get_hostname).collect()
        driver_hostname = get_hostname()[0]

        self.assertNotIn(driver_hostname, worker_hostnames,
                         "driver should be distinct from workers")
        self.assertGreater(len(set(worker_hostnames)), 1,
                           "cluster should have more than 1 workers")

