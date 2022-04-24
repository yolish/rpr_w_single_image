import torch
import torch.nn as nn
import torch.nn.functional as F
from models.posenet.PoseNet import PoseNet


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class Conv2d(nn.Module):
    """2D CONV + optionally BatchNorm and Relu

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.01, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        assert init_method in ["kaiming", "xavier"]
        self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)



def batch_dot(v1, v2):
    """
    Dot product along the dim=1
    :param v1: (torch.tensor) Nxd tensor
    :param v2: (torch.tensor) Nxd tensor
    :return: N x 1
    """
    out = torch.mul(v1, v2)
    out = torch.sum(out, dim=1, keepdim=True)
    return out

def qmult(quat_1, quat_2):
    """
    Perform quaternions multiplication
    :param quat_1: (torch.tensor) Nx4 tensor
    :param quat_2: (torch.tensor) Nx4 tensor
    :return: quaternion product
    """
    # Extracting real and virtual parts of the quaternions
    q1s, q1v = quat_1[:, :1], quat_1[:, 1:]
    q2s, q2v = quat_2[:, :1], quat_2[:, 1:]

    qs = q1s*q2s - batch_dot(q1v, q2v)
    qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) + torch.cross(q1v, q2v, dim=1)
    q = torch.cat((qs, qv), dim=1)

    return q


def qinv(q):
    """
    Inverts a unit quaternion
    :param q: (torch.tensor) Nx4 tensor (unit quaternion)
    :return: Nx4 tensor (inverse quaternion)
    """
    q_inv = torch.cat((q[:, :1], -q[:, 1:]), dim=1)
    return q_inv

class RelativePoseRegressor(PoseNet):

    def __init__(self, config, backbone_path):
        """
        Constructor
        :param backbone_path: backbone path to a resnet backbone
        """
        super(RelativePoseRegressor, self).__init__(config, backbone_path)

        # Regressor layers
        self.x_latent_fc = nn.Linear(self.backbone_dim*2, self.latent_dim)
        self.q_latent_fc = nn.Linear(self.backbone_dim*2, self.latent_dim)
        self.x_reg = nn.Linear(self.latent_dim, 3)
        self.q_reg = nn.Linear( self.latent_dim, 4)

        self.dropout = nn.Dropout(p=0.1)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward_body(self, query, ref):

        if self.backbone_type == "efficientnet":
            query_vec = self.backbone.extract_features(query)
            ref_vec = self.backbone.extract_features(ref)

        else:
            query_vec = self.backbone(query)
            ref_vec = self.backbone(ref)

        query_vec = self.avg_pooling_2d(query_vec)
        query_vec = query_vec.flatten(start_dim=1)
        ref_vec = self.avg_pooling_2d(ref_vec)
        ref_vec = ref_vec.flatten(start_dim=1)

        latent = torch.cat((query_vec, ref_vec), dim=1)

        latent_q = F.relu(self.x_latent_fc(latent))
        latent_x = F.relu(self.q_latent_fc(latent))
        return latent_x, latent_q

    def forward_heads(self, latent_x, latent_q):
        rel_x = self.x_reg(self.dropout(latent_x))
        rel_q = self.q_reg(self.dropout(latent_q))
        return rel_x, rel_q

    def forward(self, query, ref, ref_pose):
        latent_x, latent_q = self.forward_body(query, ref)
        rel_x, rel_q = self.forward_heads(latent_x, latent_q)
        x = ref_pose[:, :3] + rel_x
        q = qmult(rel_q, qinv(ref_pose[:, 3:]))
        return {"pose": torch.cat((x,q), dim=1)}


class RelativePoseRegressorLatent(nn.Module):
    """
    A class to represent a classic pose regressor (PoseNet) with an efficient-net backbone
    PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization,
    Kendall et al., 2015
    """
    def __init__(self, config):
        """
        Constructor
        """
        super(RelativePoseRegressorLatent, self).__init__()
        self.latent_dim = config.get("hidden_dim")
        self.conv1 = Conv2d(4, 16, kernel_size=3)
        self.conv2 = Conv2d(16, 64, kernel_size=3)
        self.conv_x = Conv2d(64, self.latent_dim, kernel_size=3)
        self.conv_q = Conv2d(64, self.latent_dim, kernel_size=3)

        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Regressor layers
        self.x_latent_fc = nn.Linear(self.latent_dim*4, self.latent_dim)
        self.q_latent_fc = nn.Linear(self.latent_dim*4, self.latent_dim)
        self.x_reg = nn.Linear(self.latent_dim, 3)
        self.q_reg = nn.Linear( self.latent_dim, 4)

        self.dropout = nn.Dropout(p=0.1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward_body(self, ref, latent_p_init):

        latent_ref = self.conv1(ref)
        latent_ref = self.conv2(latent_ref)
        latent_ref_x = self.conv_x(latent_ref)
        latent_ref_q = self.conv_q(latent_ref)

        latent_ref_x = self.avg_pooling_2d(latent_ref_x)
        latent_ref_x = latent_ref_x.flatten(start_dim=1)

        latent_ref_q = self.avg_pooling_2d(latent_ref_q)
        latent_ref_q = latent_ref_q.flatten(start_dim=1)

        latent_ref = torch.cat((latent_ref_x, latent_ref_q), dim=1)

        latent = torch.cat((latent_ref, latent_p_init), dim=1)

        latent_rel_x = F.relu(self.x_latent_fc(latent))
        latent_rel_q = F.relu(self.q_latent_fc(latent))
        return latent_rel_x, latent_rel_q

    def forward_heads(self, latent_rel_x, latent_rel_q):
        rel_x = self.x_reg(self.dropout(latent_rel_x))
        rel_q = self.q_reg(self.dropout(latent_rel_q))
        return rel_x, rel_q

    def forward(self, ref_rgb, ref_depth, ref_p, latent_p_init):
        ref = torch.cat((ref_rgb,ref_depth), dim=1)
        latent_rel_x, latent_rel_q = self.forward_body(ref, latent_p_init)
        rel_x, rel_q = self.forward_heads(latent_rel_x, latent_rel_q)
        x = ref_p[:, :3] + rel_x
        q = qmult(rel_q, qinv(ref_p[:, 3:]))
        return {"pose": torch.cat((x,q), dim=1)}


