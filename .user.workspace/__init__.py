#!/usr/bin/python
"""
tesseract
=========

A package for measuring the concentration of halos from Nbody simulations 
non-parametrically using Voronoi tessellation.

Subpackages
-----------
voro 
  Routines for running and manipulating data returned by the Voronoi
  tesselation routine vorovol.
nfw 
  Routines relating to fitting and determining properties of NFW profiles.
io 
  Routines for data input and output.
util
  Misc. utility routines
tests
  Routines for running and plotting different tests for the provided test halos.

"""



# Basic dependencies
import configparser
import os
import shutil

# Initialize config file
_config_file_def = os.path.join(os.path.dirname(__file__),"default_config.ini")
_config_file_usr = os.path.expanduser("~/.tessrc")
if not os.path.isfile(_config_file_usr):
    print(f'Creating user config file: {_config_file_usr}')
    shutil.copyfile(_config_file_def,_config_file_usr)


# Read config file
config_parser = configparser.ConfigParser()
config_parser.optionxform = str
config_parser.read(_config_file_def)
config_parser.read(_config_file_usr) # Overrides defaults with user options

# General options
config = {
    'outputdir': os.path.expanduser(
        config_parser.get('general', 'outputdir').strip()
    )
    if config_parser.has_option('general', 'outputdir')
    else os.getcwd()
}
# Subpackages
import voro  # noqa: E402
import nfw  # noqa: E402
import io  # noqa: E402
import util  # noqa: E402
import tests  # noqa: E402


__all__ = ['voro','nfw','io','util','tests']
