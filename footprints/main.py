# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from .training.train import TrainManager
from .evaluation.inference import InferenceManager
from .options import Options


OPTIONS = Options()
OPTIONS = OPTIONS.parse()

if OPTIONS.mode == 'train':
    print('In training mode!')
    TRAINER = TrainManager(OPTIONS)
    TRAINER.train()

elif OPTIONS.mode == 'inference':
    print('In inference mode!')
    TESTER = InferenceManager(OPTIONS)
    TESTER.run()
else:
    raise NotImplementedError
