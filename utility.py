import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from color_map import houston2018_color_map,berlin_color_map,augsburg_color_map,trento_color_map,houston2013_color_map
import logging
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler


cate=0

def draw(model,trntst_loader,H, W, dataset="Houston2018"):
    # img = np.ones((H, W, 3))
    img = np.zeros((H, W, 3))
    count = 0
    device = torch.device('cuda:0')  
    with torch.no_grad():
        for batch_idx, (hsi, lidar,hsi_pca, gt,h,w) in enumerate(trntst_loader):
            # hsi = hsi.to(device)
            # hsi = hsi[:, 0, :, :, :]
            pos = 0
            lidar = lidar.to(device)
            h = h.to(device)
            w = w.to(device)
            hsi=hsi.to(device)
            hsi_pca = hsi_pca.to(device)

            if(cate!=1):
                _,outputs = model(hsi_pca.unsqueeze(1), lidar)  #(B,C,H,W) B(B,N,C,H,W)
            else:
                _,outputs = model(hsi, lidar)

            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)

            # Houston2018 Berlin Augsburg
            for i in h:
                if dataset == "Houston2018":
                    img[i, w[pos]] = houston2018_color_map[outputs[pos]]
                elif dataset == "Berlin":
                    img[i, w[pos]] = berlin_color_map[outputs[pos]]
                elif dataset == "Trento":
                    img[i, w[pos]] = trento_color_map[outputs[pos]]
                elif dataset == "Houston2013":
                    img[i, w[pos]] = houston2013_color_map[outputs[pos]]
                else:
                    img[i, w[pos]] = augsburg_color_map[outputs[pos]]
                pos += 1
            if count == 0:
                y_pred_test = outputs
                # gty = tr_labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))  #
                # gty = np.concatenate((gty, tr_labels))
        import cv2

        i = img.copy()
        i[:, :, 0] = img[:, :, 2]
        i[:, :, 2] = img[:, :, 0]
        
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        
        cv2.imwrite(dataset + current_time +".png", i) 
    return

def draw_diff(model,trntst_loader,H, W, dataset="Houston2018"):
    img = np.ones((H, W, 3))
    count = 0
    device = torch.device('cuda:0')  
    with torch.no_grad():
        for batch_idx, (hsi, lidar,hsi_pca, gt,h,w) in enumerate(trntst_loader):
            # hsi = hsi.to(device)
            # hsi = hsi[:, 0, :, :, :]
            pos = 0
            lidar = lidar.to(device)
            h = h.to(device)
            w = w.to(device)
            hsi=hsi.to(device)
            hsi_pca = hsi_pca.to(device)
            gt=gt.to(device)

            if(cate!=1):
                _,outputs = model(hsi_pca.unsqueeze(1), lidar)  #(B,C,H,W) B(B,N,C,H,W)
            else:
                _,outputs = model(hsi, lidar)

            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)

            #for i in h:
            #    if gt[pos]!=outputs[pos]:
            #        img[i, w[pos]] = [255,0,0]
            #    else:
            #        img[i, w[pos]] = [255,255,255]
            #    pos+=1
            for i in h:
                img[i, w[pos]] = augsburg_color_map[gt[pos]]
                pos += 1
            
        import cv2
        i = img.copy()
        i[:, :, 0] = img[:, :, 2]
        i[:, :, 2] = img[:, :, 0]
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        
        cv2.imwrite(dataset + current_time +".png", i) 
    return 

def save_img(feature,gt_list,accuracy,dataset):
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=50)
    features = np.array(feature)
    features = features.reshape(features.shape[0],-1)
    features_tsne = tsne.fit_transform(features)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1],c=gt_list,s=5)
    plt.tick_params(labelsize=9)
    # plt.show()
    accuracy_str = f"{accuracy * 100:.2f}"
    filename = "TSNE/"+dataset+ f"/tsne_accuracy_{accuracy_str}.png"
    plt.savefig(filename)

def t_sne(model,test_loader,dataset):
    """Validation and get the metric
    """
    epoch_losses, epoch_accuracy = 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    houston2018_color_map = [
    [50, 205, 51],
    [173, 255, 48],
    [0, 128, 129],
    [34, 139, 34],
    [46, 79, 78],
    [139, 69, 18],
    [0, 255, 255],
    [100, 100, 100],  #255,255,255改成100
    [211, 211, 211],
    [254, 0, 0],
    [169, 169, 169],
    [105, 105, 105],
    [139, 0, 1],
    [200, 100, 0],  #####
    [254, 165, 0],
    [255, 255, 0],
    [218, 165, 33],
    [255, 0, 254],
    [0, 0, 254],
    [63, 224, 208]
    ]

    berlin_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [
            0, 204, 51], [204, 153, 52], [244, 231, 1], [204, 102, 204], [0, 53, 255]]

    augsburg_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [
            0, 204, 51], [244, 231, 1], [204, 102, 204], [0, 53, 255]]

    feature_list = []
    # gt_list = []
    count = 0
    device = torch.device('cuda:0')  
    
    with torch.no_grad():
        for batch_idx, (hsi, lidar,hsi_pca, tr_labels,h,w) in enumerate(test_loader):
            # hsi = hsi.to(device)
            # hsi = hsi[:, 0, :, :, :]
            lidar = lidar.to(device)
            hsi_pca = hsi_pca.to(device)
            tr_labels = tr_labels.to(device)

            feature,output = model(hsi_pca.unsqueeze(1),lidar)
            tr_labels = tr_labels.detach().cpu().numpy().astype(int)
            #计算一下准确率
            output = np.argmax(output.detach().cpu().numpy(), axis=1)
            accuracy = accuracy_score(tr_labels, output)
            # print(tr_labels)
            # print(type(tr_labels))
            feature = feature.detach().cpu().numpy()
            # print(feature.shape)
            # print(type(tr_labels))
            # feature_list.append(feature[0])
            if dataset == "Houston2018":
                houston2018_color_map = np.array(houston2018_color_map)
                gt=(houston2018_color_map[tr_labels]*1.0/255.0)
                save_img(feature,gt,accuracy,dataset)
            elif dataset == "Berlin":
                berlin_color_map = np.array(berlin_color_map)
                gt=(berlin_color_map[tr_labels]*1.0/255.0)
                save_img(feature,gt,accuracy,dataset)
            else: 
                augsburg_color_map = np.array(augsburg_color_map)
                gt=augsburg_color_map[tr_labels]*1.0/255.0
                save_img(feature,gt,accuracy,dataset)
                
# def TTSNE(model,test_loader,dataset):
#     """Validation and get the metric
#     """
#     epoch_losses, epoch_accuracy = 0.0, 0.0
#     criterion = torch.nn.CrossEntropyLoss()
#     houston2018_color_map = [
#     [50, 205, 51],
#     [173, 255, 48],
#     [0, 128, 129],
#     [34, 139, 34],
#     [46, 79, 78],
#     [139, 69, 18],
#     [0, 255, 255],
#     [100, 100, 100],  #255,255,255改成100
#     [211, 211, 211],
#     [254, 0, 0],
#     [169, 169, 169],
#     [105, 105, 105],
#     [139, 0, 1],
#     [200, 100, 0],  #####
#     [254, 165, 0],
#     [255, 255, 0],
#     [218, 165, 33],
#     [255, 0, 254],
#     [0, 0, 254],
#     [63, 224, 208]
#     ]

#     berlin_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [
#             0, 204, 51], [204, 153, 52], [244, 231, 1], [204, 102, 204], [0, 53, 255]]

#     augsburg_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [
#             0, 204, 51], [244, 231, 1], [204, 102, 204], [0, 53, 255]]

#     feature_list = []
#     gt_list = []
#     count = 0
#     all_outputs = []
#     all_labels = []
#     device = torch.device('cuda:0')  
#     with torch.no_grad():
#         for batch_idx, (hsi, lidar,hsi_pca, tr_labels,h,w) in enumerate(test_loader):
#             # hsi = hsi.to(device)
#             # hsi = hsi[:, 0, :, :, :]
#             lidar = lidar.to(device)
#             hsi_pca = hsi_pca.to(device)
#             tr_labels = tr_labels.to(device)
            
#             # outputs.append(output[0])
            
#             feature,output = model(hsi_pca.unsqueeze(1),lidar)
#             tr_labels = tr_labels.detach().cpu().numpy().astype(int)
#             # print(tr_labels)
#             # print(type(tr_labels))
#             output = np.argmax(output.detach().cpu().numpy(), axis=1)
#             feature = feature.detach().cpu().numpy()
            
#             all_outputs.append(output[0])
#             all_labels.append(tr_labels[0])
            
#             # print(feature.shape)
#             # print(type(tr_labels))
#             houston2018_color_map = np.array(houston2018_color_map)
#             berlin_color_map = np.array(berlin_color_map)
#             augsburg_color_map = np.array(augsburg_color_map)
#             feature_list.append(feature[0])
#             if dataset == "Houston2018":
#                 gt_list.append(houston2018_color_map[tr_labels[0]]*1.0/255.0)
#             elif dataset == "Berlin":
#                 gt_list.append(berlin_color_map[tr_labels[0]]*1.0/255.0)
#             else:
#                 gt_list.append(augsburg_color_map[tr_labels[0]]*1.0/255.0)
           
#     tsne = TSNE(n_components=2, perplexity=10, learning_rate=100)
#     features = np.array(feature_list)
#     features = features.reshape(features.shape[0],-1)
#     features_tsne = tsne.fit_transform(features)
#     plt.xlim(-100, 100)
#     plt.ylim(-100, 100)
#     plt.scatter(features_tsne[:, 0], features_tsne[:, 1],c=gt_list,s=20)
#     plt.tick_params(labelsize=9)
#     # plt.show()
#     accuracy = accuracy_score(all_labels, all_outputs)
#     accuracy_str = "tsne10" + f"{accuracy * 100:.2f}" +".png"
#     plt.savefig(accuracy_str)
#     return
 
#定义好hook方法
Mfeatures = {}
def hook_fn(module, input, output):
    # 将模块名称作为字典的键，将输出作为值存储
    Mfeatures[module.__class__.__name__] = output


# def TTSNE(model, test_loader, dataset,path):
#     """Validation and get the metric"""
#     #经过优化版 不会内存爆炸 运行更快
#     color_maps = {
#         "Houston2018": [
#             [50, 205, 51], [173, 255, 48], [0, 128, 129], [34, 139, 34], [46, 79, 78], 
#             [139, 69, 18], [0, 255, 255], [100, 100, 100], [211, 211, 211], [254, 0, 0], 
#             [169, 169, 169], [105, 105, 105], [139, 0, 1], [200, 100, 0], [254, 165, 0], 
#             [255, 255, 0], [218, 165, 33], [255, 0, 254], [0, 0, 254], [63, 224, 208]
#         ],
#         "Berlin": [
#             [26, 163, 25], [216, 216, 216], [216, 89, 89], [0, 204, 51], 
#             [204, 153, 52], [244, 231, 1], [204, 102, 204], [0, 53, 255]
#         ],
#         "Augsburg": [
#             [26, 163, 25], [216, 216, 216], [216, 89, 89], [0, 204, 51], 
#             [244, 231, 1], [204, 102, 204], [0, 53, 255]
#         ]
#     }

#     color_map = torch.tensor(color_maps.get(dataset, color_maps["Houston2018"]), dtype=torch.float32) / 255.0

#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
    
#     # 获取 Syn_layer(1) SpBh 和 MBx 的输出
#     model.layers[1].SpBh.register_forward_hook(hook_fn)
#     model.layers[1].MBx.register_forward_hook(hook_fn)
#     # 获取 FSSBlock 的两个输出
#     model.layers[1].FSSBlock.register_forward_hook(hook_fn)

#     labels=0
#     outputs=0
#     hsi_features = 0
#     x_features = 0
#     hsi_feature_eds = 0
#     x_feature_eds = 0
#     #接下来要一次性获取四个特征，使用hook方法获取特征
    
#     with torch.no_grad():
#         for batch_idx, (hsi, lidar, hsi_pca, tr_labels, h, w) in enumerate(test_loader):
#             lidar = lidar.to(device)
#             hsi_pca = hsi_pca.to(device)
#             tr_labels = tr_labels.to(device)
            
#             feature, output = model(hsi_pca.unsqueeze(1), lidar)
#             hsi_feature = Mfeatures['SpecMambaBlock']
#             x_feature = Mfeatures['MSBlock']
#             hsi_feature_ed = Mfeatures['FSSBlock'][0]
#             x_feature_ed = Mfeatures['FSSBlock'][1]
#             # feature = feature[0].unsqueeze(0)
#             hsi_feature = hsi_feature[0].unsqueeze(0)
#             x_feature = x_feature[0].unsqueeze(0)
#             hsi_feature_ed = hsi_feature_ed[0].unsqueeze(0)
#             x_feature_ed = x_feature_ed[0].unsqueeze(0)
#             label = tr_labels[0].unsqueeze(0)
#             output = torch.argmax(output[0]).unsqueeze(0)
#             #将两个特征进行堆叠
#             if(batch_idx==0):
#                 # features = feature
#                 hsi_feature = hsi_feature
#                 x_feature = x_feature
#                 hsi_feature_ed = hsi_feature_ed
#                 x_feature_ed = x_feature_ed
#                 labels = label
#                 outputs = output
#             else:
#                 # features = torch.cat((features,feature),dim=0)
#                 hsi_features = torch.cat((hsi_features,hsi_feature),dim=0)
#                 x_features = torch.cat((x_features,x_feature),dim=0)
#                 hsi_feature_eds = torch.cat((hsi_feature_eds,hsi_feature_ed),dim=0)
#                 x_feature_eds = torch.cat((x_feature_eds,x_feature_ed),dim=0)
#                 labels = torch.cat((labels,label),dim=0)
#                 outputs = torch.cat((outputs,output),dim=0)
#         #
#         perplexity=100
#         learning_rate=100
#         tsne = TSNE(n_components=2, init='pca', perplexity=perplexity, learning_rate=learning_rate)
#         # features_tsne = tsne.fit_transform(features)
#         hsi_features_tsne = tsne.fit_transform(hsi_features.reshape(hsi_features.shape[0],-1).cpu())
#         gt_colors=color_map[labels.cpu()]
#         plt.figure(figsize=(10, 10))
#         # plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) #不显示刻度
#         # # 自动调整图形范围
#         # plt.autoscale()
#         plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=gt_colors, s=20)
#         plt.xlim(-100, 100)
#         plt.ylim(-100, 100)
#         # plt.xlim(-75, 75)
#         # plt.ylim(-75, 75)
#         plt.tick_params(labelsize=9)
#         current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
#         accuracy = accuracy_score(labels.cpu(), outputs.cpu())
#         accuracy_str = path + dataset + '_' + current_time + '_' + str(perplexity) +'_' +str(learning_rate) +'_' + f"tsne_{accuracy * 100:.2f}.png"
#         plt.savefig(accuracy_str)
#         plt.close()
    
#     return accuracy



def TTSNE(model, test_loader, dataset, path):
    """Validation and get the metric"""
    # Color map initialization
    color_maps = {
        "Houston2018": [
            [50, 205, 51], [173, 255, 48], [0, 128, 129], [34, 139, 34], [46, 79, 78],
            [139, 69, 18], [0, 255, 255], [100, 100, 100], [211, 211, 211], [254, 0, 0],
            [169, 169, 169], [105, 105, 105], [139, 0, 1], [200, 100, 0], [254, 165, 0],
            [255, 255, 0], [218, 165, 33], [255, 0, 254], [0, 0, 254], [63, 224, 208]
        ],
        "Berlin": [
            [26, 163, 25], [216, 216, 216], [216, 89, 89], [0, 204, 51], 
            [204, 153, 52], [244, 231, 1], [204, 102, 204], [0, 53, 255]
        ],
        "Augsburg": [
            [26, 163, 25], [216, 216, 216], [216, 89, 89], [0, 204, 51], 
            [244, 231, 1], [204, 102, 204], [0, 53, 255]
        ]
    }

    color_map = torch.tensor(color_maps.get(dataset, color_maps["Houston2018"]), dtype=torch.float32) / 255.0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Register hooks for SpBh, MBx, and FSSBlock
    model.layers[1].SpBh.register_forward_hook(hook_fn)
    model.layers[1].MBx.register_forward_hook(hook_fn)
    model.layers[1].FSSBlock.register_forward_hook(hook_fn)

    # Initialize variables
    labels = 0
    outputs = 0
    hsi_features = 0
    x_features = 0
    hsi_feature_eds = 0
    x_feature_eds = 0
    
    # Collect features and labels
    with torch.no_grad():
        for batch_idx, (hsi, lidar, hsi_pca, tr_labels, h, w) in enumerate(test_loader):
            lidar = lidar.to(device)
            hsi_pca = hsi_pca.to(device)
            tr_labels = tr_labels.to(device)
            
            feature, output = model(hsi_pca.unsqueeze(1), lidar)
            hsi_feature = Mfeatures['SpecMambaBlock']
            x_feature = Mfeatures['MSBlock']
            hsi_feature_ed = Mfeatures['FSSBlock'][0]
            x_feature_ed = Mfeatures['FSSBlock'][1]
            
            hsi_feature = hsi_feature[0].unsqueeze(0)
            x_feature = x_feature[0].unsqueeze(0)
            hsi_feature_ed = hsi_feature_ed[0].unsqueeze(0)
            x_feature_ed = x_feature_ed[0].unsqueeze(0)
            label = tr_labels[0].unsqueeze(0)
            output = torch.argmax(output[0]).unsqueeze(0)

            if batch_idx == 0:
                hsi_features = hsi_feature
                x_features = x_feature
                hsi_feature_eds = hsi_feature_ed
                x_feature_eds = x_feature_ed
                labels = label
                outputs = output
            else:
                hsi_features = torch.cat((hsi_features, hsi_feature), dim=0)
                x_features = torch.cat((x_features, x_feature), dim=0)
                hsi_feature_eds = torch.cat((hsi_feature_eds, hsi_feature_ed), dim=0)
                x_feature_eds = torch.cat((x_feature_eds, x_feature_ed), dim=0)
                labels = torch.cat((labels, label), dim=0)
                outputs = torch.cat((outputs, output), dim=0)
    
    B,N,C,H,W = hsi_features.size()
    hsi_features = hsi_features.reshape(B,N*C,H,W)
    hsi_feature_eds = hsi_feature_eds.reshape(B,N*C,H,W)
    
    # TSNE parameters
    perplexity = 50
    learning_rate = 100
    tsne = TSNE(n_components=2, init='pca', perplexity=perplexity, learning_rate=learning_rate)

    # t-SNE and plot for each feature set
    feature_sets = {
        'hsi_features': hsi_features,
        'x_features': x_features,
        'hsi_feature_eds': hsi_feature_eds,
        'x_feature_eds': x_feature_eds
    }

    for name, features in feature_sets.items():
        scaler = MinMaxScaler()
        features = features.cpu()
        features = tsne.fit_transform(features.reshape(features.shape[0], -1))
        features = scaler.fit_transform(features.reshape(features.shape[0], -1))
        gt_colors = color_map[labels.cpu()]
        plt.figure(figsize=(10, 10))
        plt.scatter(features[:, 0], features[:, 1], c=gt_colors, s=20)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.tick_params(labelsize=9)
        
        # Save the figure with current timestamp and feature name
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        accuracy = accuracy_score(labels.cpu(), outputs.cpu())
        accuracy_str = path + dataset + '_' + current_time + '_' + name + '_' + str(perplexity) + '_' + str(learning_rate) + f"_tsne_{accuracy * 100:.2f}.png"
        plt.savefig(accuracy_str)
        plt.close()

    return accuracy


def TTSNE3D(model, test_loader, dataset, path):
    """Validation and get the metric"""
    # Color map initialization
    color_maps = {
        "Houston2018": [
            [50, 205, 51], [173, 255, 48], [0, 128, 129], [34, 139, 34], [46, 79, 78],
            [139, 69, 18], [0, 255, 255], [100, 100, 100], [211, 211, 211], [254, 0, 0],
            [169, 169, 169], [105, 105, 105], [139, 0, 1], [200, 100, 0], [254, 165, 0],
            [255, 255, 0], [218, 165, 33], [255, 0, 254], [0, 0, 254], [63, 224, 208]
        ],
        "Berlin": [
            [26, 163, 25], [216, 216, 216], [216, 89, 89], [0, 204, 51], 
            [204, 153, 52], [244, 231, 1], [204, 102, 204], [0, 53, 255]
        ],
        "Augsburg": [
            [26, 163, 25], [216, 216, 216], [216, 89, 89], [0, 204, 51], 
            [244, 231, 1], [204, 102, 204], [0, 53, 255]
        ]
    }

    color_map = torch.tensor(color_maps.get(dataset, color_maps["Houston2018"]), dtype=torch.float32) / 255.0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Register hooks for SpBh, MBx, and FSSBlock
    model.layers[1].SpBh.register_forward_hook(hook_fn)
    model.layers[1].MBx.register_forward_hook(hook_fn)
    model.layers[1].FSSBlock.register_forward_hook(hook_fn)

    # Initialize variables
    labels = 0
    outputs = 0
    hsi_features = 0
    x_features = 0
    hsi_feature_eds = 0
    x_feature_eds = 0
    
    # Collect features and labels
    with torch.no_grad():
        for batch_idx, (hsi, lidar, hsi_pca, tr_labels, h, w) in enumerate(test_loader):
            lidar = lidar.to(device)
            hsi_pca = hsi_pca.to(device)
            tr_labels = tr_labels.to(device)
            
            feature, output = model(hsi_pca.unsqueeze(1), lidar)
            hsi_feature = Mfeatures['SpecMambaBlock']
            x_feature = Mfeatures['MSBlock']
            hsi_feature_ed = Mfeatures['FSSBlock'][0]
            x_feature_ed = Mfeatures['FSSBlock'][1]
            
            hsi_feature = hsi_feature[0].unsqueeze(0)
            x_feature = x_feature[0].unsqueeze(0)
            hsi_feature_ed = hsi_feature_ed[0].unsqueeze(0)
            x_feature_ed = x_feature_ed[0].unsqueeze(0)
            label = tr_labels[0].unsqueeze(0)
            output = torch.argmax(output[0]).unsqueeze(0)

            if batch_idx == 0:
                hsi_features = hsi_feature
                x_features = x_feature
                hsi_feature_eds = hsi_feature_ed
                x_feature_eds = x_feature_ed
                labels = label
                outputs = output
            else:
                hsi_features = torch.cat((hsi_features, hsi_feature), dim=0)
                x_features = torch.cat((x_features, x_feature), dim=0)
                hsi_feature_eds = torch.cat((hsi_feature_eds, hsi_feature_ed), dim=0)
                x_feature_eds = torch.cat((x_feature_eds, x_feature_ed), dim=0)
                labels = torch.cat((labels, label), dim=0)
                outputs = torch.cat((outputs, output), dim=0)
    
    B,N,C,H,W = hsi_features.size()
    hsi_features = hsi_features.reshape(B,N*C,H,W)
    hsi_feature_eds = hsi_feature_eds.reshape(B,N*C,H,W)
    
    # TSNE parameters
    perplexity = 50
    learning_rate = 100
    tsne = TSNE(n_components=3, init='pca', perplexity=perplexity, learning_rate=learning_rate)  # 修改为 3 维

    # t-SNE and plot for each feature set
    feature_sets = {
        'hsi_features': hsi_features,
        'x_features': x_features,
        'hsi_feature_eds': hsi_feature_eds,
        'x_feature_eds': x_feature_eds
    }

    for name, features in feature_sets.items():
        features_tsne = tsne.fit_transform(features.reshape(features.shape[0], -1).cpu())
        gt_colors = color_map[labels.cpu()]

        # 创建 3D 散点图
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(features_tsne[:, 0], features_tsne[:, 1], features_tsne[:, 2], c=gt_colors, s=20)

        # 设置图形参数
        ax.set_title(f'3D t-SNE for {name}', fontsize=14)
        ax.set_xlabel('t-SNE Component 1', fontsize=10)
        ax.set_ylabel('t-SNE Component 2', fontsize=10)
        ax.set_zlabel('t-SNE Component 3', fontsize=10)
        
        # Save the figure with current timestamp and feature name
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        accuracy = accuracy_score(labels.cpu(), outputs.cpu())
        accuracy_str = path + dataset + '_' + current_time + '_' + name + '_' + str(perplexity) + '_' + str(learning_rate) + f"_3d_tsne_{accuracy * 100:.2f}.png"
        plt.savefig(accuracy_str)
        plt.close()

    return accuracy


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

# 生成 Augsburg 数据集的报告
def createAugsburgReport(net, data, device):

    # Augsburg 数据集的类别名
    augsburg_class_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Allotment', 'Commercial Area', 'Water']

    print("Augsburg Start!")
    return createReport(net, data, augsburg_class_names, device)
    print("Report Success!")
    
def createHouston2018Report(net, data, device):
    # Houston2018 数据集的类别名
    houston2018_class_names = ['Healthy grass', 'Stressed grass', 'Artificial turf', 
                               'Evergreen trees', 'Deciduous trees', 'Bare earth', 'Water', 
                               'Residential buildings', 'Non-residential buildings',
                    'Roads', 'Sidewalks', 'Crosswalks', 'Major thoroughfares', 
                    'Highways', 'Railways', 'Paved parking lots', 'Unpaved parking lots',
                    'Cars', 'Trains', 'Stadium seats']

    print("Houston2018 Start!")
    return createReport(net, data, houston2018_class_names, device)
    print("Report Success!")

# 生成 Berlin 数据集的报告
def createBerlinReport(net, data, device):
    # Berlin 数据集的类别名
    berlin_class_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil', 'Allotment', 'Commercial Area', 'Water']

    print("Berlin Start!")
    return createReport(net, data, berlin_class_names, device)
    
def getTrentoReport(net, data,device):

    trento_class_names = ['Apple trees', 'Buildings',
                          'Ground', 'Woods', 'Vineyard', 'Roads']

    print("Trento Start!")
    return createReport(net, data, trento_class_names, device)


def createHouston2013Report(net, data, device):
    # Houston2018 数据集的类别名
    houston2013_class_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water', 'Residential',
                               'Commercial', 'Road', 'Highway', 'Railway', 'Parking lot 1', 'parking lot 2', 'Tennis court', 'Running track']

    print("Houston2013 Start!")
    return createReport(net, data, houston2013_class_names, device)

def createReport(net, data, class_names, device):
    global cate
    net.eval()
    count = 0
    for hsi, x, hsi_pca, test_labels,h,w in data:
        # hsi = hsi.squeeze(1)
        hsi=hsi.cuda(device)
        hsi_pca = hsi_pca.to(device)
        x = x.to(device)
        if(cate==0):
            _ , outputs = net(hsi_pca.unsqueeze(1), x)
        else:
            _ , outputs = net(hsi, x)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred = outputs
            y_true = test_labels
            count = 1
        else:
            y_pred = np.concatenate((y_pred, outputs))
            y_true = np.concatenate((y_true, test_labels))

    classification = classification_report(
        y_true, y_pred, target_names=class_names, digits=4)
    confusion = confusion_matrix(y_true, y_pred)
    oa = accuracy_score(y_true, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_true, y_pred)

    classification = str(classification)
    confusion = str(confusion)
    oa = oa * 100
    each_acc = each_acc * 100
    aa = aa * 100
    kappa = kappa * 100

    logging.info(f'\n{classification}')
    logging.info(f'Overall accuracy (%) {oa}')
    logging.info(f'Average accuracy (%) {aa}')
    logging.info(f'Kappa accuracy (%){kappa}')
    logging.info(f'\n{confusion}')
    
    return oa,aa,kappa,each_acc

    # with open(report_path, 'w') as report:
    #     report.write('{}'.format(classification))
    #     report.write('\n')
    #     report.write('{} Overall accuracy (%)'.format(oa))
    #     report.write('\n')
    #     report.write('{} Average accuracy (%)'.format(aa))
    #     report.write('\n')
    #     report.write('{} Kappa accuracy (%)'.format(kappa))
    #     report.write('\n')
    #     report.write('\n')
    #     report.write('{}'.format(confusion))