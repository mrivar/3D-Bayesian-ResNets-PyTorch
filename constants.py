from time import strftime
import os

ROOT_PATH = {'mnist': None,
             'hmdb51': 'Data/HMDB51/',
             'infar': 'Data/InfAR_Dataset_1.0/',
             'jhmdb': 'Data/JHMDB/',
             'ucf101': 'Data/UCF101/',
             'ucf11': 'Data/UCF11/',
             'ucfsports': 'Data/ucf_sports/'}

VIDEO_PATH = {'mnist': None,
             'hmdb51': 'hmdb51_jpg/',
             'infar': 'jpg/',
             'jhmdb': 'JHMDB_jpg/',
             'ucf101': '',
             'ucf11': 'jpg/',
             'ucfsports': 'jpg/'}


ANNOTATION_PATH = {'mnist': None,
                  'hmdb51': 'hmdb51_%d.json',
                  'infar': 'infar_%d.json',
                  'jhmdb': 'jhmdb_%d.json',
                  'ucf101': None,
                  'ucf11': 'ucf11_%d.json',
                  'ucfsports': 'ucfsports_%d.json',
                  'ucf11': None}

RESULT_PATH = 'results/%s/%s_%s%d%s'
RESULT_PATH_W_COUNTER = 'results/%s_%s%d_%s__%d/'

NUM_CLASSES = {'mnist': 10,
               'hmdb51': 51,
               'infar': 12,
               'jhmdb': 21,
               'ucf101': 101,
               'ucf11': 11,
               'ucfsports': 10,
               'activitynet': 200,
               'kinetics': 400}

BAYESIAN = {'resnet': False,
            'preresnet': False,
            'wideresnet': False,
            'resnext': False,
            'densenet': False,
            'BBBresnet': True}

def create_results_dir_name(args):
  name = RESULT_PATH%(
    args.dataset,
    args.dataset,
    args.model,
    args.model_depth,
    '_q%d'%args.q_logvar_init if args.bayesian else '')
  return name

def create_results_dir_name_with_counter(args):
  counter = 0
  while True:
    counter += 1
    name = RESULT_PATH_W_COUNTER%(
      args.dataset,
      args.model,
      args.model_depth,
      strftime("%m_%d"),
      counter)
    if not os.path.isdir(name): break
  return name
