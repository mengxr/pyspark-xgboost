from ..tests import SparkDLTestCase
from sparkdl.utils import _getConfBoolean


class UtilsTest(SparkDLTestCase):

    def test_get_conf(self):
        """
        We test several default values because of pyspark.sql.RuntimeConfig being picky
        about types of conf values and default values.
        """
        # Test invalid conf value
        self.session.conf.set("myConf", "myVal")
        with self.assertRaises(Exception):
            _getConfBoolean(self.sql, "myConf")
        with self.assertRaises(Exception):
            _getConfBoolean(self.sql, "myConf", False)
        with self.assertRaises(Exception):
            _getConfBoolean(self.sql, "myConf", "False")

        # Test valid conf value
        self.session.conf.set("myConf", "False")
        self.assertFalse(_getConfBoolean(self.sql, "myConf", "True"))
        self.session.conf.set("myConf", True)
        self.assertTrue(_getConfBoolean(self.sql, "myConf", "False"))

        # Test unset conf value
        self.session.conf.unset("myConf")
        self.assertFalse(_getConfBoolean(self.sql, "myConf", "False"))
