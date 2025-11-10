import torch
import torch.nn as nn

# 假设您有图像特征和点云特征的张量数据，分别表示为 image_features 和 point_cloud_features

# 假设您已经有了图像特征和点云特征的张量数据，分别表示为 image_features 和 point_cloud_features

# 定义 NCE Loss
class NCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(NCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, positive, negative):
        # 计算正样本与负样本之间的相似度分数
        positive_scores = torch.matmul(features, positive.t()) / self.temperature
        negative_scores = torch.matmul(features, negative.t()) / self.temperature
        
        # 计算对数概率
        positive_log_probs = torch.log(torch.exp(positive_scores).sum(dim=1))
        negative_log_probs = torch.log(torch.exp(negative_scores).sum(dim=1))
        
        # 计算损失
        loss = - (positive_log_probs - negative_log_probs).mean()

        return loss
