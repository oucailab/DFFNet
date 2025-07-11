#以berlin数据集为准
import torch  
import thop
import time
# from model.ClassifierNet import Net, Bottleneck
from model.net import Net
from setting.options import opt
import yaml

#S2ENet
# model = S2ENet(input_channels=180, input_channels2=4, n_classes=7,patch_size=11).cuda()  # 定义你的PyTorch模型  

#DFINet
# model = DFINet(channel_hsi=180, channel_msi=4, class_num=7).cuda()  # 定义你的PyTorch模型 

#AsyFFNet
# model = model.ClassifierNet.Net(180,  4, 192, model.ClassifierNet.Bottleneck, 2, 2, 8, 0.002).cuda()

#ExVit
# model = MViT(patch_size = 11,num_patches = [180,4],num_classes = 8,dim = 256,
# depth = 10,heads = 4,mlp_dim = 32,dropout = 0.1,emb_dropout = 0.1,mode = 'MViT'
# ).cuda()

#HCT
# model = HCTnet(in_channels=4,num_classes=7).cuda()

# MACN
# model = MixConvNet(in_channels=4,num_classes=7).cuda()

#首先读取一个window_size
with open('dataset_info.yaml', 'r') as file:
    data = yaml.safe_load(file)
data=data[opt.dataset]
window_size=data['window_size']
pca_num=data['pca_num']
slar_channel_num=data['slar_channel_num']
num_classes=data['num_classes']
# #MSFMamba
torch.cuda.empty_cache()
model = Net(opt.dataset).cuda()

#FusatNet
# model = FusAtNet(input_channels=244, input_channels2=4, num_classes=8).cuda()
print(torch.cuda.memory_allocated())
print(torch.cuda.max_memory_allocated())


input1 = (1, 1, pca_num, window_size, window_size)  # 输入大小，这里假设是一个批量的3通道224x224图像,如果要经过pca处理需要修改一下  
input2 = (1, slar_channel_num, window_size, window_size) 
start_time = time.time()
flops, params = thop.profile(model, inputs=(torch.randn(input1).cuda(),torch.randn(input2).cuda()))  
end_time = time.time()
# 计算运行时间
elapsed_time = end_time - start_time
print(f"Params: {params / 1e6} M")  # 打印参数量（以百万为单位）
print(f"FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）  
print(f"运行时间: {elapsed_time:.4f} 秒")
print(torch.cuda.memory_allocated())
print(torch.cuda.max_memory_allocated())