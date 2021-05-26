from __future__ import division
from __future__ import absolute_import
import torch


class ReactiveBaseline():
    def __init__(self, params, update_rate):
        self.params = params
        self.update_rate = update_rate
        self.value = torch.zeros(1)
        if self.params["use_cuda"]:
            self.value = self.value.cuda()

    def get_baseline_value(self):
        return self.value

    def update(self, target):
        self.value = torch.add((1 - self.update_rate) * self.value, self.update_rate * target)
