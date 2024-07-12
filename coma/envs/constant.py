# Environmental parameter definitions
NUM_EDGE_CORES = 10
NUM_EDGE_CLOCK_FREQUENCY = 4
NUM_CLOUD_CORES = 54
NUM_CLOUD_CLOCK_FREQUENCY = 4


COST_TYPE = 1e4

# Channel type
LTE = 1
WIFI = 2
BT = 3
NFC = 4
WIRED = 5

# Vehicle type definition
SPEECH_RECOGNITION = 1
NLP = 2
FACE_RECOGNITION = 3
SEARCH_REQ = 4
LANGUAGE_TRANSLATION = 5
PROC_3D_GAME = 6
VR = 7
AR = 8

# Unit of data size
BYTE = 8
KB = 1024 * BYTE
MB = 1024 * KB
GB = 1024 * MB
TB = 1024 * GB
PB = 1024 * TB

# CPU clock frequency unit
KHZ = 1e3
MHZ = KHZ * 1e3
GHZ = MHZ * 1e3

#  Data transfer rate unit
KBPS = 1e3
MBPS = KBPS * 1e3
GBPS = MBPS * 1e3

# time unit
MS = 1e-3

'''
arrival rate            Mbps
arrival data size       Mbps
time slot interval      sec (TBD)
Edge computation cap.   3.3*10^2~10^4
'''

# Maximum number of shards
MAX_SHARD_NUM = 128