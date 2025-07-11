import torch
import torch.nn as nn
import yaml
from modules.DynamicBlock import DynamicFilterBlock
from modules.fusion import SpectralSpatialAdaptiveFusionBlock

class MultiStageDFFMBlock(nn.Module):
    def __init__(self, channels,
                 dfb_params, ssafb_params, ssafb_out_channels):
        super().__init__()
        self.channels = channels

        self.dfb_hsi = DynamicFilterBlock(dim=channels,  **dfb_params)
        self.dfb_sar = DynamicFilterBlock(dim=channels,**dfb_params)

        self.ssafb = SpectralSpatialAdaptiveFusionBlock(
            hsi_channels=channels,
            sar_channels=channels,
            out_channels=ssafb_out_channels,
            **ssafb_params
        )

        self.adjust_ssafb_out = nn.Conv2d(ssafb_out_channels, channels, kernel_size=1)


    def forward(self, hsi_feat, sar_feat):
        h_dfb_out = self.dfb_hsi(hsi_feat)
        s_dfb_out = self.dfb_sar(sar_feat)

        ssafb_out = self.ssafb(hsi_feat, sar_feat)
        ssafb_out_adjusted = self.adjust_ssafb_out(ssafb_out)

        hsi_next_feat = h_dfb_out + ssafb_out_adjusted
        sar_next_feat = s_dfb_out + ssafb_out_adjusted

        return hsi_next_feat, sar_next_feat


class Net(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        with open('dataset_info.yaml', 'r') as file:
            config = yaml.safe_load(file)

        data_params = config[dataset]

        self.out_features = data_params['num_classes']
        img_size = data_params['window_size']
        pca_num_initial = data_params['pca_num']
        sar_num_initial = data_params['slar_channel_num']

        dffm_stage_channels = data_params.get('dffm_stage_channels', 64)
        #默认为3个块
        num_dffm_stages = data_params.get('num_dffm_stages',3)

        hsi_stream_conv_out_channels = data_params.get('intermediate_conv_j_out_channels', dffm_stage_channels)
        sar_stream_conv_out_channels = data_params.get('intermediate_conv_k_out_channels', dffm_stage_channels)

        hsi_3d_conv_output_maps = dffm_stage_channels // 4
        target_spectral_dim_after_3d_conv = 4
       
        kernel_d_hsi = pca_num_initial - target_spectral_dim_after_3d_conv + 1
       
        if kernel_d_hsi <= 0:
            kernel_d_hsi = pca_num_initial
            target_spectral_dim_after_3d_conv = 1
        if pca_num_initial < target_spectral_dim_after_3d_conv:
            target_spectral_dim_after_3d_conv = max(1, pca_num_initial)

        self.hsi_initial_3d_conv = nn.Sequential(
            nn.Conv3d(1, hsi_3d_conv_output_maps, kernel_size=(kernel_d_hsi, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(hsi_3d_conv_output_maps), nn.ReLU(inplace=True)
        )
        hsi_flattened_channels = hsi_3d_conv_output_maps * target_spectral_dim_after_3d_conv
       
        # 总是添加1x1卷积进行调整
        self.hsi_channel_adjust_conv = nn.Conv2d(hsi_flattened_channels, dffm_stage_channels, kernel_size=1)


        # SAR/LiDAR 2D 卷积
        self.sar_initial_2d_conv = nn.Sequential(
            nn.Conv2d(sar_num_initial, dffm_stage_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(dffm_stage_channels), nn.ReLU(inplace=True)
        )

        # 多级 DFFM 模块
        self.dffm_stages = nn.ModuleList()
        for i in range(num_dffm_stages):
            dfb_params_stage = {
                'num_filter_bases': data_params.get(f'dfb_num_bases_s{i}', data_params.get('dfb_num_bases', 4)),
                'mlp_ratio': data_params.get(f'dfb_mlp_ratio_s{i}', data_params.get('dfb_mlp_ratio', 2)),
                'bias': data_params.get(f'dfb_bias_s{i}', data_params.get('dfb_bias', False))
            }
            ssafb_in_dffm_out_channels = data_params.get(f'ssafb_in_dffm_out_channels_s{i}', dffm_stage_channels)
            ssafb_params_stage = {
                'shuffle_groups': data_params.get(f'ssafb_shuffle_groups_s{i}', data_params.get('ssafb_shuffle_groups', 2)),
                'reduction_ratio_ca': data_params.get(f'ssafb_ca_reduction_s{i}', data_params.get('ssafb_ca_reduction', 16)),
                'kernel_size_sa': data_params.get(f'ssafb_sa_kernel_s{i}', data_params.get('ssafb_sa_kernel', 5))
            }
            self.dffm_stages.append(
                MultiStageDFFMBlock(
                    channels=dffm_stage_channels,
                    dfb_params=dfb_params_stage,
                    ssafb_params=ssafb_params_stage,
                    ssafb_out_channels=ssafb_in_dffm_out_channels
                )
            )

        # HSI 流的最终卷积
        self.hsi_final_conv = nn.Sequential(
            nn.Conv2d(dffm_stage_channels, hsi_stream_conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hsi_stream_conv_out_channels), nn.ReLU(inplace=True)
        )

        # SAR 流的最终卷积
        self.sar_final_conv = nn.Sequential(
            nn.Conv2d(dffm_stage_channels, sar_stream_conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(sar_stream_conv_out_channels), nn.ReLU(inplace=True)
        )

        # 最终融合前的 SAR 流调整
        self.sar_fusion_adjust_conv = nn.Conv2d(sar_stream_conv_out_channels, hsi_stream_conv_out_channels, kernel_size=1)

        # 分类器
        self.final_feature_dim_flat = hsi_stream_conv_out_channels * (img_size ** 2)
        classifier_hidden_dim = data_params.get('classifier_hidden_dim', 256)
        self.classifier = nn.Sequential(
            nn.Linear(self.final_feature_dim_flat, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(data_params.get('dropout_classifier', 0.5)),
            nn.Linear(classifier_hidden_dim, self.out_features)
        )

    def forward(self, hsi_input, sar_input):
        # HSI 3D 卷积处理
        hsi_3d_features = self.hsi_initial_3d_conv(hsi_input)
        B, C, N, H, W = hsi_3d_features.shape
        hsi_features_flattened = hsi_3d_features.view(B , C*N, H, W)
        hsi_features_processed = self.hsi_channel_adjust_conv(hsi_features_flattened)

        # SAR 2D 卷积处理
        sar_features_processed = self.sar_initial_2d_conv(sar_input)

        # 通过多级 DFFM
        current_hsi_features = hsi_features_processed
        current_sar_features = sar_features_processed
        for dffm_block in self.dffm_stages:
            current_hsi_features, current_sar_features = dffm_block(current_hsi_features, current_sar_features)

        # HSI流最终处理
        hsi_stream_output = self.hsi_final_conv(current_hsi_features)
        # SAR流最终处理
        sar_stream_output = self.sar_final_conv(current_sar_features)
        sar_stream_output_adjusted = self.sar_fusion_adjust_conv(sar_stream_output)

        # 最终加法融合
        fused_features_for_classification = hsi_stream_output + sar_stream_output_adjusted

        # Refinement & Classification
        flattened_features_for_classifier = fused_features_for_classification.view(B, -1)
       
        logits = self.classifier(flattened_features_for_classifier)

        return fused_features_for_classification, logits