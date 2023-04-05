"""
Create a dist_info directory
As defined in the wheel specification
"""

import os
import shutil
import sys
import warnings
from contextlib import contextmanager
from distutils import log
from distutils.core import Command
from pathlib import Path

from .. import _normalization
from .._deprecation_warning import SetuptoolsDeprecationWarning


class dist_info(Command):
    """
    This command is private and reserved for internal use of setuptools,
    users should rely on ``setuptools.build_meta`` APIs.
    """

    description = "DO NOT CALL DIRECTLY, INTERNAL ONLY: create .dist-info directory"

    user_options = [
        ('egg-base=', 'e', "directory containing .egg-info directories"
                           " (default: top of the source tree)"
                           " DEPRECATED: use --output-dir."),
        ('output-dir=', 'o', "directory inside of which the .dist-info will be"
                             "created (default: top of the source tree)"),
        ('tag-date', 'd', "Add date stamp (e.g. 20050528) to version number"),
        ('tag-build=', 'b', "Specify explicit tag to add to version number"),
        ('no-date', 'D', "Don't include date stamp [default]"),
        ('keep-egg-info', None, "*TRANSITIONAL* will be removed in the future"),
    ]

    boolean_options = ['tag-date', 'keep-egg-info']
    negative_opt = {'no-date': 'tag-date'}

    def initialize_options(self):
        self.egg_base = None
        self.output_dir = None
        self.name = None
        self.dist_info_dir = None
        self.tag_date = None
        self.tag_build = None
        self.keep_egg_info = False

    def finalize_options(self):
        if self.egg_base:
            msg = "--egg-base is deprecated for dist_info command. Use --output-dir."
            warnings.warn(msg, SetuptoolsDeprecationWarning)
            self.output_dir = self.egg_base or self.output_dir

        dist = self.distribution
        project_dir = dist.src_root or os.curdir
        self.output_dir = Path(self.output_dir or project_dir)

        egg_info = self.reinitialize_command("egg_info")
        egg_info.egg_base = str(self.output_dir)

        if self.tag_date:
            egg_info.tag_date = self.tag_date
        else:
            self.tag_date = egg_info.tag_date

        if self.tag_build:
            egg_info.tag_build = self.tag_build
        else:
            self.tag_build = egg_info.tag_build

        egg_info.finalize_options()
        self.egg_info = egg_info

        name = _normalization.safer_name(dist.get_name())
        version = _normalization.safer_best_effort_version(dist.get_version())
        self.name = f"{name}-{version}"
        self.dist_info_dir = os.path.join(self.output_dir, f"{self.name}.dist-info")

    @contextmanager
    def _maybe_bkp_dir(self, dir_path: str, requires_bkp: bool):
        if requires_bkp:
            bkp_name = f"{dir_path}.__bkp__"
            _rm(bkp_name, ignore_errors=True)
            _copy(dir_path, bkp_name, dirs_exist_ok=True, symlinks=True)
            try:
                yield
            finally:
                _rm(dir_path, ignore_errors=True)
                shutil.move(bkp_name, dir_path)
        else:
            yield

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.egg_info.run()
        egg_info_dir = self.egg_info.egg_info
        assert os.path.isdir(egg_info_dir), ".egg-info dir should have been created"

        log.info("creating '{}'".format(os.path.abspath(self.dist_info_dir)))
        bdist_wheel = self.get_finalized_command('bdist_wheel')

        # TODO: if bdist_wheel if merged into setuptools, just add "keep_egg_info" there
        with self._maybe_bkp_dir(egg_info_dir, self.keep_egg_info):
            bdist_wheel.egg2dist(egg_info_dir, self.dist_info_dir)


def _rm(dir_name, **opts):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name, **opts)


def _copy(src, dst, **opts):
    if sys.version_info < (3, 8):
        opts.pop("dirs_exist_ok", None)
    shutil.copytree(src, dst, **opts)
