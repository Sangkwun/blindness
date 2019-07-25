from torch import nn
import torch
import numpy as np

class HierarchLoss(nn.Module):

    def __init__(self, method='h_classification'):
        assert method in ['h_classification', 'h_regression']
        super(HierarchLoss, self).__init__()

        self.method = method

    def forward(self, pred, target):
        if self.method == 'h_classification':
            num_class = pred.size(1)

            bin_pred = pred[:, 0]
            mul_pred = pred[:, 1:]

            cvt_bin_target = target != 0
            cvt_mul_target = torch.cat((torch.unsqueeze(torch.zeros(num_class - 1), 0), torch.eye(num_class - 1)))
            cvt_mul_target = cvt_mul_target[target]

            binary_loss = nn.BCEWithLogitsLoss()
            binary_loss_result = binary_loss(bin_pred, cvt_bin_target.type(torch.float).cuda())
            mul_loss = nn.BCEWithLogitsLoss()
            mul_loss_result = mul_loss(mul_pred, cvt_mul_target.type(torch.float).cuda())

            return binary_loss_result + mul_loss_result
        elif self.method == 'h_regression':
            print('Not yet')

def decode_label(pred):
    bin_pred = pred[:, 0]
    mul_pred = pred[:, 1:]
    decode_pred = torch.argmax(mul_pred, dim=1) + 1
    decode_pred[nn.functional.sigmoid(bin_pred) < 0.5] = 0
    return decode_pred
    
def decode_label_np(pred):
    pred = pred.values
    bin_pred = pred[0]
    mul_pred = pred[1:]
    decode_pred = np.argmax(mul_pred, axis=0) + 1
    if sigmoid(bin_pred) < 0.5:
        decode_pred = 0
    return decode_pred

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
