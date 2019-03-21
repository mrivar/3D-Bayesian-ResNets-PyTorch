from time import strftime
from os import makedirs
from os.path import isdir

ROOT_PATH = {'mnist': None,
             'infar': 'Data/InfAR_Dataset_1.0/',
             'hmdb51': 'Data/HMDB51/',
             'ucf101': 'Data/UCF101/'}

VIDEO_PATH = {'mnist': None,
             'infar': 'Data/InfAR_Dataset_1.0/',
             'hmdb51': 'hmdb51_jpg/',
             'ucf101': 'Data/UCF101/'}

ANNOTATION_PATH = {'mnist': None,
                  'infar': None,
                  'hmdb51': 'hmdb51_%d.json',
                  'ucf101': None}

RESULT_PATH = 'results/%s_%s%d_%s__%d/'

NUM_CLASSES = {'mnist': 10,
               'infar': 12,
               'hmdb51': 51,
               'ucf101': 101,
               'activitynet': 200,
               'kinetics': 400}

BAYESIAN = {'resnet': False,
            'preresnet': False,
            'wideresnet': False,
            'resnext': False,
            'densenet': False,
            'BBBresnet': True}

def create_results_dir_name(args):
  counter = 0
  while True:
    counter += 1
    name = RESULT_PATH%(
      args.dataset,
      args.model,
      args.model_depth,
      strftime("%m_%d"),
      counter)
    if not isdir(name): break
  makedirs(name)
  return name
