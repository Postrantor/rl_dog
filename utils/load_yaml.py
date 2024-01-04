import os
from ruamel.yaml import YAML


def load_parameters(file_path):
  # 获取当前脚本的路径
  # script_path = os.path.dirname(os.path.realpath(__file__))
  # yaml_file = os.path.join(script_path, file_path)
  with open(file_path, 'r', encoding='utf-8') as f:
    yaml = YAML(typ='safe')  # default, if not specfied, is 'rt' (round-trip)
    parameters = yaml.load(f)
  return parameters
