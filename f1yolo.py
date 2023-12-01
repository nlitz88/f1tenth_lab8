"""Contains the PyTorch Implementation of the F1Tenth YOLO model."""

import torch
from torch import nn

class F110_YOLO(torch.nn.Module):
    def __init__(self):
        super(F110_YOLO, self).__init__()
        # TODO: Change the channel depth of each layer
        # First, going to try to sort of mimic what they do in the original YOLO paper.
        conv1_num_filters = 192
        self.conv1 = nn.Conv2d(3, conv1_num_filters, kernel_size = 7, padding = 1, stride = 2)
        self.batchnorm1 = nn.BatchNorm2d(conv1_num_filters)
        self.relu1 = nn.ReLU(inplace = True)

        conv2_num_filters = 256
        self.conv2 = nn.Conv2d(conv1_num_filters, conv2_num_filters, kernel_size = 3, padding = 1, stride = 2)
        self.batchnorm2 = nn.BatchNorm2d(conv2_num_filters)
        self.relu2 = nn.ReLU(inplace = True)

        conv3_num_filters = 512
        self.conv3 = nn.Conv2d(conv2_num_filters, conv3_num_filters, kernel_size = 3, padding = 1, stride = 2)
        self.batchnorm3 = nn.BatchNorm2d(conv3_num_filters)
        self.relu3 = nn.ReLU(inplace = True)

        conv4_num_filters = 1024
        self.conv4 = nn.Conv2d(conv3_num_filters, conv4_num_filters, kernel_size = 3, padding = 1, stride = 2)
        self.batchnorm4 = nn.BatchNorm2d(conv4_num_filters)
        self.relu4 = nn.ReLU(inplace = True)

        conv5_num_filters = 1024
        self.conv5 = nn.Conv2d(conv4_num_filters, conv5_num_filters, kernel_size = 3, padding = 1, stride = 2)
        self.batchnorm5 = nn.BatchNorm2d(conv5_num_filters)
        self.relu5 = nn.ReLU(inplace = True)

        conv6_num_filters = 1024
        self.conv6 = nn.Conv2d(conv5_num_filters, conv6_num_filters, kernel_size = 3, padding = 1, stride = 1)
        self.batchnorm6 = nn.BatchNorm2d(conv6_num_filters)
        self.relu6 = nn.ReLU(inplace = True)

        conv7_num_filters = 1024
        self.conv7 = nn.ConvTranspose2d(conv6_num_filters, conv7_num_filters, kernel_size = 3, padding = 1, stride = 1)
        self.batchnorm7 = nn.BatchNorm2d(conv7_num_filters)
        self.relu7 = nn.ReLU(inplace = True)

        conv8_num_filters = 1024
        self.conv8 = nn.ConvTranspose2d(conv7_num_filters, conv8_num_filters, kernel_size = 3, padding = 1, stride = 1)
        self.batchnorm8 = nn.BatchNorm2d(conv8_num_filters)
        self.relu8 = nn.ReLU(inplace = True)

        self.conv9 = nn.Conv2d(conv8_num_filters, 5, kernel_size = 1, padding = 0, stride = 1)
        self.relu9 = nn.ReLU()

    def forward(self, x):
        debug = 0 # change this to 1 if you want to check network dimensions
        if debug == 1: print(0, x.shape)
        x = torch.relu(self.batchnorm1(self.conv1(x)))
        if debug == 1: print(1, x.shape)
        x = torch.relu(self.batchnorm2(self.conv2(x)))
        if debug == 1: print(2, x.shape)
        x = torch.relu(self.batchnorm3(self.conv3(x)))
        if debug == 1: print(3, x.shape)
        x = torch.relu(self.batchnorm4(self.conv4(x)))
        if debug == 1: print(4, x.shape)
        x = torch.relu(self.batchnorm5(self.conv5(x)))
        if debug == 1: print(5, x.shape)
        x = torch.relu(self.batchnorm6(self.conv6(x)))
        if debug == 1: print(6, x.shape)
        x = torch.relu(self.batchnorm7(self.conv7(x)))
        if debug == 1: print(7, x.shape)
        x = torch.relu(self.batchnorm8(self.conv8(x)))
        if debug == 1: print(8, x.shape)
        x = self.conv9(x)
        if debug == 1: print(9, x.shape)
        x = torch.cat([x[:, 0:3, :, :], torch.sigmoid(x[:, 3:5, :, :])], dim=1)

        return x

    def get_loss(self, result, truth, lambda_coord = 5, lambda_noobj = 1):
        # Looks like there's a discrepency between the output shape from the network and the ground truth shape.
        x_loss = (result[:, 1, :, :] - truth[:, 1, :, :]) ** 2
        y_loss = (result[:, 2, :, :] - truth[:, 2, :, :]) ** 2
        w_loss = (torch.sqrt(result[:, 3, :, :]) - torch.sqrt(truth[:, 3, :, :])) ** 2
        h_loss = (torch.sqrt(result[:, 4, :, :]) - torch.sqrt(truth[:, 4, :, :])) ** 2
        class_loss_obj = truth[:, 0, :, :] * (truth[:, 0, :, :] - result[:, 0, :, :]) ** 2
        class_loss_noobj = (1 - truth[:, 0, :, :]) * lambda_noobj * (truth[:, 0, :, :] - result[:, 0, :, :]) ** 2

        total_loss = torch.sum(lambda_coord * truth[:, 0, :, :] * (x_loss + y_loss + w_loss + h_loss) + class_loss_obj + class_loss_noobj)

        return total_loss