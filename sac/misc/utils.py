import datetime
import dateutil.tz
import os.path

import sac.rllab.misc.logger as logger

PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))


def timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f-%Z')


def initialize_logger(**kwargs):
    tabular_log_file = os.path.join(
        kwargs.get('log_dir'), kwargs.get('exp_name'), 'progress.csv')
    logger.add_tabular_output(tabular_log_file)

    logger.set_snapshot_mode(kwargs.get('snapshot_mode'))
    logger.set_snapshot_gap(kwargs.get('snapshot_gap'))
    logger.push_prefix("[%s] " % kwargs.get('exp_name'))
