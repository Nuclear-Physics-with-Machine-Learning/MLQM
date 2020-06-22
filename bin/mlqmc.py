import sys, os
import pathlib
import configparser

import argparse



################

# Add the local folder to the import path:
mlqm_dir = os.path.dirname(os.path.abspath(__file__))
mlqm_dir = os.path.dirname(mlqm_dir)
sys.path.insert(0,mlqm_dir)


class exec(object):

    def __init__(self):

        # This technique is taken from: https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
        parser = argparse.ArgumentParser(
            description='Run ML Based QMC Calculations',
            usage='python exec.py config.ini [<args>]')

        parser.add_argument("-c", "--config-file",
            type        = pathlib.Path,
            required    = True,
            help        = "Python configuration file to describe this run.")

        args = parser.parse_args()


        # Open the config file:
        config = configparser.ConfigParser()
        config.read(args.config_file)

        print(config.sections())


    def build_optimization()

    # def build_walkers(self):




if __name__ == "__main__":
    e = exec()