# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from .train import Trainer
from .inference import Tester
from .options import Options


OPTIONS = Options()
OPTIONS = OPTIONS.parse()

if OPTIONS.mode == 'train':
    print('In training mode!')
    TRAINER = Trainer(OPTIONS)
    TRAINER.train()

elif OPTIONS.mode == 'inference':
    print('In inference mode!')
    TESTER = Tester(OPTIONS)
    TESTER.test()
else:
    raise NotImplementedError
