#!/usr/bin/env python3

#  Copyright (c) 2022-2023 Mira Geoscience Ltd.
#
#  This file is part of my_app package.
#
#  All rights reserved.

"""
Creates locked environment files for Conda to install the application within the environment.

Usage: from the conda base environment, at the root of the project:
> python devtools/create_application_env_files.py

To prepare the conda base environment, see devtools/setup-conda-base.bat
"""

from __future__ import annotations

import argparse
import re
import subprocess
import warnings
from pathlib import Path

from add_url_tag_sha256 import computeSha256
from run_conda_lock import per_platform_env

_archive_ext = ".tar.gz"

app_name = "my-app"


def create_standalone_lock(git_url: str, extras=[], suffix=""):
    print(
        f"# Creating lock file for stand-alone environment (extras={','.join(extras)})..."
    )
    py_ver = "3.10"
    platform = "win-64"
    base_filename = f"conda-py-{py_ver}-{platform}{suffix}"
    initial_lock_file = Path(f"environments/{base_filename}-tmp.lock.yml")
    try:
        per_platform_env(py_ver, extras, suffix=f"{suffix}-tmp")
        final_lock_file = Path(f"{base_filename}.lock.yml")
        assert initial_lock_file.exists()
        add_application(git_url, initial_lock_file, final_lock_file)
    finally:
        print("# Cleaning up intermediate files ...")
        initial_lock_file.unlink()
        for f in Path("environments").glob("conda-py-*-tmp.lock.yml"):
            f.unlink()


def add_application(git_url: str, lock_file: Path, output_file: Path):
    print(f"# Patching {lock_file} for standalone environment ...")
    pip_dependency_re = re.compile(r"^\s*- (geoh5py|simpeg|simpeg-archive) @")
    pip_dependency_lines = []
    with open(lock_file) as input:
        for line in input:
            if pip_dependency_re.match(line):
                pip_dependency_lines.append(line)

    pip_section_re = re.compile(r"^\s*- pip:\s*$")
    application_pip = f"    - {app_name} @ {git_url}\n"
    print(f"# Patched file: {output_file}")
    with open(output_file, "w") as patched:
        with open(lock_file) as input:
            for line in input:
                if not pip_dependency_re.match(line):
                    patched.write(line)
                if pip_section_re.match(line):
                    for pip_line in pip_dependency_lines:
                        patched.write(pip_line)
                    patched.write(application_pip)


def git_url_with_ref(args) -> tuple[str, str]:
    assert args.repo_url
    if args.ref_type == "sha":
        ref = args.ref
    elif args.ref_type == "tag":
        ref = f"refs/tags/{args.ref}"
    elif args.ref_type == "branch":
        ref = f"refs/heads/{args.ref}"
    else:
        raise RuntimeError(f"Unhandled reference type ${args.ref_type}")
    return args.repo_url, ref


def build_git_url(repo_url: str, ref: str) -> str:
    return f"{repo_url}/archive/{ref}{_archive_ext}"


def get_git_url():
    process = subprocess.run(
        ["git", "config", "--get-regexp", "remote.*.url"],
        check=True,
        capture_output=True,
        text=True,
    )

    mira_remote_re = re.compile(r".*\bgithub.com(?::|/)(MiraGeoscience/\S+)\s*$")
    for line in process.stdout.splitlines():
        match = mira_remote_re.match(line)
        if match:
            segment = match[1][:-4] if match[1].endswith(".git") else match[1]
            return f"https://github.com/{segment}"
        warnings.warn(
            "Could not detect the remote MiraGeoscience github repository for this application."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Creates locked environment files for Conda to install application within the environment."
    )
    parser.add_argument("ref_type", choices=["sha", "tag", "branch"])
    parser.add_argument(
        "ref", help="the git commit reference for the application pip dependency"
    )
    parser.add_argument(
        "--url",
        dest="repo_url",
        default=get_git_url(),
        help="the URL of the git repo for the application pip dependency",
    )

    repo_url, ref_path = git_url_with_ref(parser.parse_args())
    basename_match = re.match(r".*/([^/]*)$", repo_url)
    assert basename_match
    basename = basename_match[1]
    git_download_url = build_git_url(repo_url, ref_path)
    checksum = computeSha256(git_download_url, basename)
    checked_git_url = f"{git_download_url}#sha256={checksum}"

    create_standalone_lock(checked_git_url)


if __name__ == "__main__":
    main()
