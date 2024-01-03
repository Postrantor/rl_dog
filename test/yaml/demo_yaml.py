import yaml
import os
from ruamel.yaml import YAML


def load_parameters_from_yaml(file_path):
  # 获取当前脚本的路径
  script_path = os.path.dirname(os.path.realpath(__file__))
  # 拼接相对路径和文件名
  yaml_file = os.path.join(script_path, file_path)
  with open(yaml_file, 'r', encoding='utf-8') as f:
    yaml = YAML(typ='safe')  # default, if not specfied, is 'rt' (round-trip)
    parameters = yaml.load(f)
    # parameters = yaml.safe_load(f)
  return parameters


def run_algorithm(parameters):
  # 在这里调用你的算法程序，并传入参数
  # 例如：
  # result = your_algorithm(parameters['param1'], parameters['param2'])
  print('%d, %d', parameters['training']['drift_weight'],
        parameters['training']['drift_weight'])
  # pass


if __name__ == '__main__':
  # 读取 YAML 文件中的参数
  parameters = load_parameters_from_yaml('parameters.yaml')

  # 运行算法程序
  run_algorithm(parameters)
