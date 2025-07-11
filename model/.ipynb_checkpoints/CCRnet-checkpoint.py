import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

class CCRnet(nn.Module):
    def __init__(self, x1_num, x2_num, out_num, window_size):
        super(CCRnet, self).__init__()
        self.x1_num = x1_num
        self.x2_num = x2_num
        self.out_num = out_num
        self.window_size = window_size

        # HSI分支
        self.x1_conv1 = nn.Conv2d(x1_num, 16, kernel_size=3, padding=1)
        self.x1_bn1 = nn.BatchNorm2d(16)
        self.x1_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.x1_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.x1_bn2 = nn.BatchNorm2d(32)
        self.x1_pool2 = nn.AdaptiveAvgPool2d((1, 1))  # 确保输出为1x1

        # SAR分支
        self.x2_conv1 = nn.Conv2d(x2_num, 16, kernel_size=3, padding=1)
        self.x2_bn1 = nn.BatchNorm2d(16)
        self.x2_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.x2_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.x2_bn2 = nn.BatchNorm2d(32)
        self.x2_pool2 = nn.AdaptiveAvgPool2d((1, 1))  # 确保输出为1x1

        # 全连接层输入维度修正为32+32=64
        self.x_en_fc1 = nn.Linear(32 + 32, 64)
        self.x_en_bn1 = nn.BatchNorm1d(64)
        self.x_en_fc2 = nn.Linear(64, 32)
        self.x_en_bn2 = nn.BatchNorm1d(32)
        self.x_en_fc3 = nn.Linear(32, 32)
        self.x_en_bn3 = nn.BatchNorm1d(32)
        
        self.x_en_fc4 = nn.Linear(32, out_num)

        self._initialize_weights()

    # 初始化权重保持不变
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        # HSI分支
        x1 = x1.view(-1, self.x1_num, self.window_size, self.window_size)
        x1 = F.relu(self.x1_bn1(self.x1_conv1(x1)))
        x1 = self.x1_pool1(x1)
        x1 = F.relu(self.x1_bn2(self.x1_conv2(x1)))
        x1 = self.x1_pool2(x1)  # 输出[B,32,1,1]
        x1 = torch.flatten(x1, 1)  # [B,32]

        # SAR分支
        x2 = x2.view(-1, self.x2_num, self.window_size, self.window_size)
        x2 = F.relu(self.x2_bn1(self.x2_conv1(x2)))
        x2 = self.x2_pool1(x2)
        x2 = F.relu(self.x2_bn2(self.x2_conv2(x2)))
        x2 = self.x2_pool2(x2)  # 输出[B,32,1,1]
        x2 = torch.flatten(x2, 1)  # [B,32]

        joint = torch.cat([x1, x2], dim=1)  # [B,64]

        x_en = F.relu(self.x_en_bn1(self.x_en_fc1(joint)))
        x_en = F.relu(self.x_en_bn2(self.x_en_fc2(x_en)))
        x_en = F.relu(self.x_en_bn3(self.x_en_fc3(x_en)))
        out = self.x_en_fc4(x_en)
        return out

class Net(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        with open('dataset_info.yaml', 'r') as file:
            data = yaml.safe_load(file)
        data = data[dataset]
        
        pca_num = data['pca_num']
        slar_channel_num = data['slar_channel_num']
        num_classes = data['num_classes']
        window_size=data['window_size']
        
        self.CCR = CCRnet(
            x1_num=pca_num, 
            x2_num=slar_channel_num, 
            out_num=num_classes,
            window_size=window_size
        )
        
    def forward(self, hsi, sar):
        # 移除多余的维度（如果存在）
        hsi = hsi.squeeze(dim=1)  # 假设输入是 [B,1,30,7,7] → [B,30,7,7]
        return None, self.CCR(hsi, sar)