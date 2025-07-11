# from model.build_model import Net
from setting.dataLoader import get_loader
from setting.options import opt
import torch
from setting.utils import clip_gradient, adjust_lr, compute_accuracy
import yaml
import utility
# from model.ExVit import MViT
# from model.MixConvNet_houston2013 import MixConvNet
# from model.HCTnet import HCTnet
from model.FreqLC import Net
# from model.S2ENet import S2ENet
from setting.utils import create_folder, random_seed_setting
# from model.DFINet import Net

random_seed_setting(6)

#首先读取一个window_size
with open('dataset_info.yaml', 'r') as file:
    data = yaml.safe_load(file)
data=data[opt.dataset]
window_size=data['window_size']
H=data['image_h']
W=data['image_w']

# test_loader, test_num = get_loader(opt.dataset,opt.batchsize,window_size,
#                         category='test',shuffle=False,num_workers=opt.num_work)

# trntst_loader, trntst_num = get_loader(opt.dataset,opt.batchsize,window_size,
#                         category='train_test',shuffle=False,num_workers=opt.num_work)

train_loader,test_loader,trntst_loader,all_loader,train_num,val_num,trntst_num=get_loader(dataset=opt.dataset, 
        batchsize=opt.batchsize,num_workers=opt.num_work, pin_memory=True)

# model = HCTnet(in_channels=4,num_classes=7).cuda()
model = Net(opt.dataset).cuda()
# model = S2ENet(input_channels=180, input_channels2=4, n_classes=7).cuda()
# model = MViT(
#     patch_size = 11,
#     num_patches = [244,4],
#     num_classes = 8,
#     dim = 64,
#     depth = 6,
#     heads = 4,
#     mlp_dim = 32,
#     dropout = 0.1,
#     emb_dropout = 0.1,
#     mode = 'MViT'
# ).cuda()
#model = MixConvNet(in_channels=1,num_classes=6).cuda()
# model = S2ENet(input_channels=244, input_channels2=4, n_classes=8).cuda()
# model = Net(channel_hsi=244, channel_msi=4, class_num=8).cuda()
# model.load_state_dict(torch.load('checkpoints/DFI/BerlinDFI_epoch_6.pth'))
# model.load_state_dict(torch.load('checkpoints/MSFMamba/2024-07-02_16-40-17_78.32421008869179_Berlin_Net_epoch_4.pth'))
model.load_state_dict(torch.load('/root/autodl-tmp/sspc-dev/checkpoints/Augsburg/weight/2025-03-21_16-04-05_92.60038191577209_Augsburg_Net_epoch_31.pth'))
# model.load_state_dict(torch.load('checkpoints/MSFMamba/Berlin_Net_epoch_40.pth'))
# 测一下berlin数据集的tsne


model.cuda()
# criterion = torch.nn.CrossEntropyLoss()

model.eval()





import time

start_time = time.perf_counter()

with torch.no_grad():
    # utility.createHouston2018Report(model, trntst_loader, 'checkpoints/'+opt.dataset+".txt", device='cuda:0')
    # utility.createBerlinReport(model, test_loader, 'checkpoints/'+opt.dataset+".txt", device='cuda:0')
    utility.draw(model, trntst_loader, H, W, dataset=opt.dataset)
# utility.draw(model,trntst_loader,H, W, dataset="Berlin")
    # utility.TTSNE(model,test_loader,'Houston2018')
    # utility.TTSNE(model,test_loader,opt.dataset)

end_time = time.perf_counter()
print(f"运行时间：{end_time - start_time}秒")

# iteration = len(test_loader)
# with torch.no_grad():
#     loss_sum = 0
#     acc=0
#     num=0
#     for i, (hsi, Xdata,hsi_pca, gt) in enumerate(test_loader, start=1):
#         hsi = hsi.cuda()
#         Xdata = Xdata.cuda()
#         hsi_pca = hsi_pca.cuda()
#         gt = gt.cuda()
#         outputs = model(hsi_pca.unsqueeze(1),Xdata)
#         acc += compute_accuracy(outputs,gt)*len(gt)
#         num += len(gt)
#         loss = criterion(outputs,gt)
#         loss_sum += loss.detach()
#         if(i%100==0):
#             print('已经处理了'+str(i)+"个batch的图像")
 #     acc_avg = acc / num
#     print('准确率为'+str(acc_avg))