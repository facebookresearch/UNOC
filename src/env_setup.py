# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from glob import glob

if "PATH_DATA" not in os.environ:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'parsers')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'npybvh')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'third-party')))
    # Set up a virtual monitor for EGL
    os.system('Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &')

    PATH_DATA_PREFIX = "/root/dataset/nnvrik"
    os.environ["PATH_UNOC"] = os.path.join(PATH_DATA_PREFIX, "UNOC")
    os.environ["PATH_DATA"] = os.path.join(PATH_DATA_PREFIX, "train_data")

    # Convert datasets into training data records
    if not os.path.exists(os.environ["PATH_DATA"]):
        os.makedirs(os.environ["PATH_DATA"])

    # Prepare training data records
    parsers = []
    if len(glob(os.path.join(os.environ["PATH_UNOC"], '*.npy'))) == 0:
        from unoc_parser import UnocParser

        parsers.append(UnocParser())

    for parser in parsers:
        print('Preparing Training Data Records: {}'.format(parser.root_path))
        # for animations with correct scale
        parser.convert_all_files(normalize_skeleton=None)
        # for animations with normalized scale, all 3 parsers use the same base skeleton from UNOC
        parser.convert_all_files(normalize_skeleton=parser.get_normalized_path())
