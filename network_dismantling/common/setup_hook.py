import logging

from network_dismantling._setup_hook import setup_hook

folder = "network_dismantling/common/external_dismantlers/"


@setup_hook
def setup(*args,
          logger=logging.getLogger("dummy"),
          **kwargs):
    from subprocess import check_output

    cd_cmd = f"cd {folder} && "
    cmd = "make clean && make"

    try:
        logger.info(check_output(cd_cmd + cmd, shell=True, text=True))
    except Exception as e:
        raise e
