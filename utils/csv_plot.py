# -*- coding:utf-8 -*-

# %%
import os
import time
# Math
import pandas as pd
import numpy as np


# %%
class CSVClass(object):
  def mk_dir(self, path):
    path = path.strip()  # 去除首位空格
    path = path.rstrip("\\")  # 去除尾部 \ 符号
    isExists = os.path.exists(path)  # 判断路径是否存在
    if not isExists:  # 如果不存在则创建目录
      os.makedirs(path)  # 创建目录操作函数
      print(path + ' 创建成功')
      return path
    else:  # 如果目录存在则不创建，并提示目录已存在
      print(path + ' 目录已存在')
      return path

  def file_list(self, file_dir):
    '''读取目录内的所有文件，并返回一个列表'''
    arg_in = []
    for root, dirs, files in os.walk(file_dir):
      for files_list in files:
        arg_in.append(root + '/' + files_list)
    return arg_in

  def path_save(self, folder='', suffix='', num=0):
    '''以日期为文件名依据，存储在相对路径下'''
    time_str = time.localtime()
    time_stamp = ''.join(str(i) for i in time_str[:])  # time_str[:6]
    fname = self.mk_dir("{}/{}{}{}".format(
        folder, time_str[0], time_str[1], time_str[2])) + "/{}_{}.{}".format(
            time_stamp, num, suffix)
    return fname

  def save2csv(self, header, data):
    '''@ref.[数据按列写入csv]'''
    write = {
        header[0]: data[0],
        header[1]: data[1],
        header[2]: data[2],
        header[3]: data[3],
    }
    pd.DataFrame(write).to_csv(self.path_save('csv', 'csv'))

  def csv2plot(self, path_csv):
    '''
    @ref.[numpy读取csv文件]
    与 file_list() 连用，获取文件列表
    '''
    file = open(path_csv, 'r')
    try:
      data = np.loadtxt(
          file,
          dtype=float,
          delimiter=",",
          skiprows=1,
          usecols=(1, 2, 3, 4),
      )
    finally:
      file.close()
    return data


# %%
'''
@ref.[numpy读取csv文件](https://blog.csdn.net/u012413551/article/details/87890989)
@ref.[数据按列写入csv](https://www.jianshu.com/p/9fa558be86bd)
@ref.[对numpy数据多个维度上等间隔取值](https://blog.csdn.net/qq_41381865/article/details/107795217)
'''
