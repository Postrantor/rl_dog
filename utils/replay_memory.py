# -*- coding: utf-8 -*-

class ReplayMemory():
  """
  @brief: 有限列表对象
  @detail: 这段代码定义了一个有限列表对象 `ReplayMemory`，它用于保存和管理强化学习中的状态转换。这个类具有以下方法：
    - `__init__(self, capacity)`: 类的构造函数，初始化有限列表对象。参数 `capacity` 表示列表的容量。
    - `push(self, state, action, next_state, reward, done)`: 将一个状态转换保存到内存中。参数包括当前状态 `state`、应用的动作 `action`、应用动作后的下一个状态 `next_state`、应用动作获得的奖励 `reward`，以及布尔值 `done`，表示是否达到终止状态。
    - `sample(self, batch_size)`: 返回一个指定大小的内存样本。参数 `batch_size` 表示要返回的内存项数。
    - `__len__(self)`: 返回内存的长度。

  这个类的目的是处理强化学习算法中的回放记忆功能。在强化学习中，回放记忆用于保存之前的状态转换，供训练过程中进行样本的随机抽样。通过将状态转换保存到内存中，并从中随机选择一定数量的样本，可以使训练过程收敛更快并提高模型的性能。
  """

  def __init__(self, capacity):
    """
    @brief: 构造函数: 初始化有限列表对象
    @param: capacity: 列表的容量
    """
    self.capacity = int(capacity)
    self.memory = []
    self.position = 0

  def push(self, state, action, next_state, reward, done):
    """
    @brief: 保存状态转换到列表中
    @param: state: 环境中的当前状态
    @param: action: 应用到环境的动作
    @param: next_state: 应用动作后的下一个状态
    @param: reward: 应用动作获得的奖励
    @param: done: 布尔值，表示是否达到了终止状态
    @return: None
    """
    # 如果当前列表大小小于容量，则在内存中添加一个新位置
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = (state, action, next_state, reward, done)
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    """
    @brief: 返回指定大小的内存随机样本
    @param: batch_size: 要返回的内存项数
    @return: 回放内存的一个样本
    """
    if batch_size >= self.capacity:
      batch = self.memory
    else:
      batch = random.sample(self.memory, batch_size)
    sample = map(np.stack, zip(*batch))
    return sample

  def __len__(self):
    """
    @brief: 返回内存的长度
    @return: 内存的长度
    """
    return len(self.memory)
