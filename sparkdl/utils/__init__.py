import sys
import logging


def _getConfBoolean(sqlContext, key, defaultValue):
    """
    Get the conf "key" from the given sqlContext,
    or return the default value if the conf is not set.
    This expects the conf value to be a boolean or string; if the value is a string,
    this checks for all capitalization patterns of "true" and "false" to match Scala.
    :param key: string for conf name
    """
    # Convert default value to str to avoid a Spark 2.3.1 + Python 3 bug: SPARK-25397
    val = sqlContext.getConf(key, str(defaultValue))
    # Convert val to str to handle unicode issues across Python 2 and 3.
    lowercase_val = str(val.lower())
    if lowercase_val == 'true':
        return True
    elif lowercase_val == 'false':
        return False
    else:
        raise Exception("_getConfBoolean expected a boolean conf value but found value of type {} "
                        "with value: {}".format(type(val), val))


def get_logger(name, level='INFO'):
    """ Gets a logger by name, or creates and configures it for the first time. """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # If the logger is configured, skip the configure
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
    return logger


def _get_max_num_concurrent_tasks(sc):
    """Gets the current max number of concurrent tasks."""
    # spark 3.1 and above has a different API for fetching max concurrent tasks
    if sc._jsc.sc().version() >= '3.1':
        return sc._jsc.sc().maxNumConcurrentTasks(
            sc._jsc.sc().resourceProfileManager().resourceProfileFromId(0)
        )
    return sc._jsc.sc().maxNumConcurrentTasks()
