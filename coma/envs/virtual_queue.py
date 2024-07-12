import logging
logger = logging.getLogger(__name__)
import numpy as np
import uuid
import collections

class VirtualQueue(object):
    def __init__(self, veh_app_types=None, max_resource=np.inf):
        self.uuid = uuid.uuid4()
        self.resources = collections.OrderedDict()
        self.max_resource = max_resource
        self.veh_app_types = veh_app_types
        self.length = 0
        self.arrival_size_buffer = ResourceBuffer(max_size=100)

    def __del__(self):
        if len(self.resources.keys()):
            ids = list(self.resources.keys())
            for id in ids:
                del self.resources[id]
        del self.arrival_size_buffer
        del self

    def set_length(self, length):
        self.length = length

    def get_length(self):
        return self.length

    def allocation_resource(self, resources, alloc_timestamp):
        resources_sum = sum([re for resource in resources for re in resource])

        resources = [[row[i] for row in resources] for i in range(len(resources[0]))]

        self.arrival_size_buffer.add((alloc_timestamp, resources_sum))
        self.resources = dict(zip(list(self.veh_app_types), resources))
    def get_arrival_buffer(self):
        return self.arrival_size_buffer

class ResourceBuffer:
    def __init__(self, max_size=100):
        self.storage = list()
        self.max_size = max_size

    def add(self, resource):
        self.storage.append(resource)
        if len(self.storage) > self.max_size:
            self.storage = self.storage[1:]
        else:
            pass

    def get_buffer(self):
        return self.storage[::-1]

    def get_last_obs(self):
        if self.storage:
            return self.storage[:-1]
        else:
            return None