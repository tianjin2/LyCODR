
import numpy as np
# 导入一个生成唯一标识符的库
import uuid
# 从sac.envs.constants模块导入所有常量
from coma.envs.constant import *

class Channel:
    def __init__(self, channel_type, fading=None, rate=None):
        self.uuid = uuid.uuid4()
        self.channel_type = channel_type
        self.bw = []  # Initialise bandwidth list
        self.max_coverage = []  # Initialise the maximum coverage list
        self.fading = fading  # Initialise channel fading properties

        # If no specific uplink or downlink rate is provided, set the default rate based on the channel type
        if not rate:
            if channel_type == LTE:
                self.up = 75 * MBPS
                self.down = 300 * MBPS
            elif channel_type == WIFI:
                self.up = 135 * MBPS
                self.down = 135 * MBPS
            elif channel_type == BT:
                self.up = 22 * MBPS
                self.down = 22 * MBPS
            elif channel_type == NFC:
                self.up = 212 * KBPS
                self.down = 212 * KBPS
            else:  # Wired channel type by default
                self.up = 200 * KBPS
                self.down = 200 * KBPS
        else:
            self.up = rate[0]
            self.down = rate[1]

    def get_uuid(self):
        return self.uuid.hex

    def get_channel_type(self):
        return self.channel_type

    def get_rate(self, is_up=True):
        if is_up:
            mean_rate = self.up
        else:
            mean_rate = self.down


        if not self.fading:
            return mean_rate
        elif self.fading in 'rR':
            return np.random.rayleigh(np.sqrt(2 / np.pi) * mean_rate)