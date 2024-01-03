import yaml


def load_parameters_from_yaml(file_path):
  with open(file_path, 'r') as f:
    parameters = yaml.safe_load(f)
  return parameters


def run_algorithm(parameters):
  # 在这里调用你的算法程序，并传入参数
  # 例如：
  # result = your_algorithm(parameters['param1'], parameters['param2'])
  print('%d, %d', parameters['app_name']['param1'], parameters['app_name']['param2'])
  # pass


if __name__ == '__main__':
  # 读取 YAML 文件中的参数
  parameters = load_parameters_from_yaml('parameters.yaml')

  # 运行算法程序
  run_algorithm(parameters)
