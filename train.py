import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


import time
import os
import logging
from datetime import datetime
import torch
import torch.nn.functional as F
from setting.dataLoader import get_loader
from setting.utils import clip_gradient, adjust_lr, compute_accuracy
from setting.utils import create_folder, random_seed_setting
from setting.loss import FocalLoss
from setting.options import opt
from model.net import Net

import yaml
import numpy as np
import utility

# from lion_pytorch import Lion

random_seed_setting(6)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
GPU_NUMS = torch.cuda.device_count()

# # Logs
save_path = create_folder(opt.save_path + opt.dataset)
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

logging.basicConfig(filename=save_path + '/log/' + opt.dataset+current_time+ 'log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO,
                    filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')

#输出配置信息
##################
with open('dataset_info.yaml', 'r') as file:
    data = yaml.safe_load(file)
data=data[opt.dataset]
for key, value in data.items():
    print(key, ':', value)
    logging.info(f'{key}:{value}')
#################

logging.info(f'********************start train!********************')
logging.info(f'Config--epoch:{opt.epoch}; lr:{opt.lr}; batch_size:{opt.batchsize};')
# writer_train = SummaryWriter(save_path + 'summary'+'/train')
# writer_test = SummaryWriter(save_path + 'summary'+'/test')

# load data
train_loader,test_loader,trntst_loader,all_loader,train_num,val_num,trntst_num=get_loader(dataset=opt.dataset, 
        batchsize=opt.batchsize,num_workers=opt.num_work,useval=opt.useval, pin_memory=True)

print(f'Loading data, including {train_num} training images and {val_num} \
        validation images and {trntst_num} train_test images')
logging.info(f'Loading data, including {train_num} training images and {val_num} \
        validation images and {trntst_num} train_test images')
# model
#model=torch.compile(Net(opt.dataset)).cuda()
model =Net(opt.dataset).cuda()

if GPU_NUMS == 1:
    print(f"Loading model, and using single GPU - {opt.gpu_id}")
elif GPU_NUMS > 1:
    print(f"Loading model, and using multiple GPUs - {opt.gpu_id}")
    model = torch.nn.DataParallel(model)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"number of params: {n_parameters}")
# check model size
if not os.path.exists('module_size'):
    os.makedirs('module_size')
for name, module in model.named_children():
    torch.save(module, 'module_size/' + '%s' % name + '.pth')
# optimizer
optimizer = torch.optim.Adam(model.parameters(), opt.lr)
# optimizer = Lion(model.parameters(), lr=opt.lr, weight_decay=1e-2)#可以还真能训练 效果还差了

# Loss function
# criterion = FocalLoss().cuda()
# criterion = torch.nn.CrossEntropyLoss(reduction="none").cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
# L2loss=torch.nn.MSELoss().cuda()


# Restore training from checkpoints
if(opt.start_epoch!=1):
    optimizer.load_state_dict(torch.load(opt.optimizer))
    model.load_state_dict(torch.load(opt.model))


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    model.train()
    loss_all = 0
    iteration = len(train_loader)
    acc=0
    num=0
    for i, (hsi, Xdata, hsi_pca, gt,h,w) in enumerate(train_loader, start=1):#虽然感觉冗余但是可以统一书写
        #现在只需要加载hsi_pca即可
        optimizer.zero_grad()
        hsi = hsi.cuda()
        Xdata = Xdata.cuda()
        hsi_pca = hsi_pca.cuda()
        gt = gt.cuda()
        _,outputs = model(hsi_pca.unsqueeze(1),Xdata) #这里改了一下
        gt_loss = criterion(outputs,gt) #计算损失函数
        loss = gt_loss
        loss.backward()
        optimizer.step()
        loss_all += loss.detach()
        acc += compute_accuracy(outputs,gt)*len(gt)
        num += len(gt)
    loss_avg = loss_all / iteration
    acc_avg = acc / num
    print(f'Epoch [{epoch:03d}/{opt.epoch:03d}]:Loss_train_avg={loss_avg:.4f}')
    print(acc_avg)
    logging.info(f'Epoch [{epoch:03d}/{opt.epoch:03d}], Loss_train_avg: {loss_avg:.4f},acc_avg:{acc_avg:.4f}')
    # writer_train.add_scalar('Loss-avg', loss_avg, global_step=epoch)
    # writer_train.add_scalar('acc_avg', acc_avg, global_step=epoch)
    if(epoch==opt.epoch or epoch==opt.epoch//2):
        torch.save(optimizer.state_dict(),
                save_path+'/weight/' + current_time + opt.dataset + "_optimizer" + "Epoch" + str(epoch) + '.pth')
        torch.save(model.state_dict(), save_path+'/weight/' + current_time + opt.dataset + '_Net_epoch_{}.pth'.format(epoch))

# test function  需要记得修改
best_acc = opt.best_acc
best_epoch = opt.best_epoch

def test(val_loader, model, epoch, save_path):
    global best_acc, best_epoch
    if(opt.dataset=='Berlin'):
        oa,aa,kappa,acc=utility.createBerlinReport(net=model, data=val_loader,device='cuda:0')
    elif(opt.dataset=='Houston2018'):
        oa,aa,kappa,acc=utility.createHouston2018Report(net=model, data=val_loader, device='cuda:0')
    elif(opt.dataset=='Trento'):
        oa,aa,kappa,acc=utility.getTrentoReport(net=model, data=val_loader, device='cuda:0')
    elif(opt.dataset=='Houston2013'):
        oa,aa,kappa,acc=utility.createHouston2013Report(net=model, data=val_loader, device='cuda:0')
    else:
        oa,aa,kappa,acc=utility.createAugsburgReport(net=model, data=val_loader,device='cuda:0')
    
        # writer_test.add_scalar('Loss-avg', loss_epoch, global_step=epoch)
        # writer_test.add_scalar('acc_avg', acc_avg, global_step=epoch)
    if oa > best_acc:   #保存最高的准确率
        best_acc, best_epoch = oa, epoch
        if(epoch>=1):  #每一轮都保存
            # torch.save(optimizer.state_dict(),
            #     save_path + opt.dataset + "_optimizer" + "Epoch" + str(epoch) + '.pth')
            torch.save(optimizer.state_dict(),
                save_path+'/weight/' + current_time + '_' + str(best_acc) +'_' + opt.dataset + "_optimizer" + "Epoch" + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_path+'/weight/' + current_time + '_' + str(best_acc) +'_' + opt.dataset + '_Net_epoch_{}.pth'.format(epoch))
    print(f'Epoch [{epoch:03d}/{opt.epoch:03d}]'
            f' best_acc={best_acc:.4f}, Best_epoch:{best_epoch:03d}')
    logging.info(f'Best_acc:{best_acc:.4f},Best_epoch:{best_epoch:03d}')


if __name__ == '__main__':
    print("-------------------Config-------------------\n"
          f'epoch:\t\t{opt.epoch}\n'
          f'lr:\t\t{opt.lr}\n'
          f'batchsize:\t{opt.batchsize}\n'
          f'decay_epoch:\t{opt.decay_epoch}\n'
          f'decay_rate:\t{opt.decay_rate}\n'
          "--------------------------------------------\n")
    print("Start train...")
    time_begin = time.time()

    for epoch in range(opt.start_epoch, opt.epoch + 1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)

        # writer_train.add_scalar('learning-rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        # if(epoch==opt.epoch):
        test(test_loader, model, epoch, save_path)
        time_epoch = time.time()
        print(f"Time out:{time_epoch - time_begin:.2f}s\n")
        logging.info(f"Time out:{time_epoch - time_begin:.2f}s\n")