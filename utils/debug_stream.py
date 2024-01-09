# -*- coding:utf-8 -*-

import time


class DebugSteam(object):

  def print_log(self, log='time_msg', node_id=0):
    '''
        print('Serial Port is OK{}'.format(self.print_log('time_msg')))
    '''
    # ----- PROTECTED REGION ID(InitialDeviceClass.print_log) ENABLED START -----#
    if log == 'full_msg':
      msg = '\n\t- PortID: {}\n\t- NodeID: {}\n\t- Buad: {}\n\t- Timeout: {}\n\t- Time: {}'.format(self.dev_serial.port, node_id, self.dev_serial.baudrate, self.dev_serial.timeout, time.asctime())
    elif log == 'time_msg':
      msg = '\n\t- PortID: {}\n\t- NodeID: {}\n\t- Time: {}'.format(self.dev_serial.port, node_id, time.asctime())
    else:
      msg = '\n\t- NodeID: {}'.format(time.asctime())
    return msg
    # ----- PROTECTED REGION END -----#	//	CopleyControl.print_log

  def debug_stream(self, *msg):
    # ----- PROTECTED REGION ID(InitialDeviceClass.debug_stream) ENABLED START -----#
    strs = ''
    for target_tuple in reversed(msg):
      strs = str(target_tuple) + '\t' + strs
    print(strs + '{}'.format(self.print_log('', node_id='debug')))
    print('--- --- ---')
    # ----- PROTECTED REGION END -----#	//	CopleyControl.debug_stream
