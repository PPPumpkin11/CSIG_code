import torch
from torch import nn
from torch.autograd import Variable

import torch.nn.functional as F
# classes

import torch
from torch import nn

from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

# helpers
class SE_ContextGating(nn.Module):
    def __init__(self, vlad_dim, hidden_size, drop_rate=0.1, gating_reduction=8):
        super(SE_ContextGating, self).__init__()

        self.fc1 = nn.Linear(vlad_dim, hidden_size)
        self.dropout = nn.Dropout(drop_rate)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.gate = torch.nn.Sequential(
            nn.Linear(hidden_size, hidden_size // gating_reduction),
            nn.BatchNorm1d(hidden_size // gating_reduction),
            nn.ReLU(),

            nn.Linear(hidden_size // gating_reduction, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bn1(self.dropout(self.fc1(x)))
        gate = self.gate(x)
        activation = x * gate
        return activation
class NextVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, ActionLength=60,num_clusters=64,pose_dim=24 * 6, expansion=2, group=8, dim=2048, num_class=10):
        super(NextVLAD, self).__init__()

        self.num_clusters = num_clusters
        self.expansion = expansion
        self.group = group
        self.dim = dim
        assert pose_dim == 24 * 6
        self.pose_embedding = nn.Linear(pose_dim, dim)
        # self.bn1 = nn.BatchNorm1d(group * num_clusters)
        # self.bn2 = nn.BatchNorm1d(num_clusters * expansion * dim // group)
        self.centroids1 = nn.Parameter(torch.rand(expansion * dim, group * num_clusters))
        self.centroids2 = nn.Parameter(torch.rand(1, expansion * dim // group, num_clusters))
        self.fc1 = nn.Linear(dim, expansion * dim)
        self.fc2 = nn.Linear(dim * expansion, group)

        self.cg = SE_ContextGating(num_clusters * expansion * dim // group, dim)
        self.fc3 = nn.Linear(dim, num_class)

    def forward(self, x):  # 2,4,2048,1,1
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(-1, 60, 24 * 6)
        x = self.pose_embedding(x) #Nx60x(24*6)->Nx60x2048
        # x = torch.relu(x)
        # print(x.size())
        # dd
        max_frames = x.size(1)
        x = x.view(x.size()[:3])  # 2,4,2048
        # print(x.size())
        # dd
        # x_3d = F.normalize(x, p=2, dim=2)  # across descriptor dim, torch.Size([2,4, 2048, 1, 1])
        # print(x_3d.size())
        # dd
        # x_ = x
        vlads = []
        for t in range(x.size(0)):
            x_ = x[t, :, :]  # 4,2048
            x_ = self.fc1(x_)  # expand, 4,2*2048

            # attention
            attention = torch.sigmoid(self.fc2(x_))  # 4,8
            attention = attention.view(-1, max_frames * self.group, 1)

            feature_size = self.expansion * self.dim // self.group
            # reshaped_input = tf.reshape(input, [-1, self.expansion * self.feature_size])
            reshaped_input = x_.view(-1, self.expansion * self.dim)  # 4,2*2048
            # activation = tf.matmul(reshaped_input, cluster_weights)
            activation = torch.mm(reshaped_input, self.centroids1)  # 4,8*32
            # activation = self.bn1(activation)
            # activation = tf.reshape(activation, [-1, self.max_frames * self.groups, self.cluster_size])
            activation = activation.view(-1, max_frames * self.group, self.num_clusters)  # 1,32,32
            # activation = tf.nn.softmax(activation, axis=-1)
            activation = F.softmax(activation, dim=-1)  # 1,32,32
            # activation = tf.multiply(activation, attention)
            activation = activation * attention  # 1,32,32
            # a_sum = tf.sum(activation, -2, keep_dims=True)
            a_sum = activation.sum(dim=-2, keepdim=True)  # 1,32,1

            # a = tf.multiply(a_sum, cluster_weights2)
            a = a_sum * self.centroids2  # 1,512,32 (512=dim*expansion//group,32=clusters)
            # activation = tf.transpose(activation, perm=[0, 2, 1])
            activation = activation.permute(0, 2, 1)  # 1,32,1
            # reshaped_input = tf.reshape(input, [-1, self.max_frames * self.groups, feature_size])
            reshaped_input = x_.view(-1, max_frames * self.group, feature_size)  # 1,32,512
            vlad = torch.bmm(activation, reshaped_input)  # 1,32,512
            # vlad = tf.transpose(vlad, perm=[0, 2, 1])
            vlad = vlad.permute(0, 2, 1)
            # vlad = tf.subtract(vlad, a)
            vlad = vlad - a  # 1,512,32
            # vlad = tf.nn.l2_normalize(vlad, 1)
            # vlad = F.normalize(vlad, p=2, dim=1)
            # print(vlad.size())
            # print(self.num_clusters,feature_size)

            # vlad = tf.reshape(vlad, [-1, self.cluster_size * feature_size])
            # vlad = vlad.view(1,self.num_clusters * feature_size)  # [1, 16384]
            vlad = vlad.reshape(self.num_clusters * feature_size)  # [1, 16384]
            # print(vlad.size())
            # dd
            vlads.append(vlad)
        vlads = torch.stack(vlads, dim=0)
        # vlads = self.bn2(vlads)  # [2, 16384]

        x = self.cg(vlads)  # SE Context Gating
        x = self.fc3(x)

        return x

class LSTM3V1(nn.Module):
    def __init__(self,ActionLength=60,hidden_dim1=1024,hidden_dim2=1024,dim_fc=24*6,mean_dim=1,
                 input_channel=3,num_classes=14,drop=0., aux_logits=True, transform_input=False):
        super(LSTM3V1, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        ##########att setting##########
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        # self.seq_len = ActionLength
        self.use_gpu = torch.cuda.is_available()
        self.dim_fc = dim_fc
        self.lstm_input_dim = self.dim_fc
        # self.sign = LBSign.apply
        self.lstm_7a = nn.LSTM(self.dim_fc, hidden_dim1,dropout=drop,num_layers=3,batch_first=True)
        # self.lstm_7b = nn.LSTM(hidden_dim1, hidden_dim2,batch_first=True)
        self.classifier = nn.Linear(hidden_dim2, num_classes)
        # self.linear_att_RL=self._make_layers_att_RL(self.dim_fc+hidden_dim1+hidden_dim2,)
    def init_hidden(self,hidden_dim,num_layer=1,seq_num=None):
        if seq_num is None:
            seq_num= self.seq_num*self.num_att
        if self.use_gpu:
            h0 = Variable(torch.zeros(num_layer, seq_num, hidden_dim).cuda())
            c0 = Variable(torch.zeros(num_layer, seq_num, hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(num_layer, seq_num, hidden_dim))
            c0 = Variable(torch.zeros(num_layer,seq_num, hidden_dim))
        return (h0, c0)
    def forward(self, x0):
        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()
        vid_num =x0.size(0)
        vid_len =x0.size(1)
        #################
        conv_out_7a_sep = x0#out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        hidden_a= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1,num_layer=3)
        # hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        lstm_out_b_, hidden_b = self.lstm_7a(conv_out_7a_sep, hidden_a)
        lstm_out_base = lstm_out_b_[:,-1,:]#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        return x_out_base
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        # return self.fn(self.norm(x), **kwargs)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, ActionLength=60,
    pose_dim=24 * 6,
    num_classes = 14,
    # num_classes = 10,
    dim = 1024,
    # depth = 6,
    depth = 12,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1, dim_head=64, pool='cls'):
        super(ViT,self).__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert ActionLength == 60
        assert pose_dim == 24 * 6

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.Linear(patch_dim, dim),
        # )
        self.pose_embedding = nn.Linear(pose_dim, dim)
        # self.pos_embedding = nn.Parameter(torch.randn(1, ActionLength + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, fea_npy):
        x = fea_npy.permute(0, 3, 1, 2).contiguous()
        x = x.view(-1, 60, 24 * 6)
        x = self.pose_embedding(x)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        # dummpy = torch.ones(x.size(0),1).cuda()
        # x = torch.cat((x,dummpy),dim=-1)
        # print(x.size())
        # dd
        return x
        # return self.mlp_head(x)
import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


##
import numpy as np

def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A

import sys
import numpy as np

# sys.path.extend(['../'])
# from graph import tools

num_node = 24 #20
self_link = [(i, i) for i in range(num_node)]
# inward_ori_index = [(1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
#                     (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19),(20,21),(21,22),(22,23),(23,24)]
inward_ori_index = [(23,21),(21,19),(19,17),(17,14),(14,9),
                    (11,8),(8,5),(5,2),(2,0),(0,3),(3,6),(6,9),
                    (10,7),(7,4),(4,1),(1,0),(0,3),(3,6),(6,9),
                    (15,12),(12,9)]
inward_ori_index = [(i +1, j +1) for (i, j) in inward_ori_index]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # if in_channels == 3 or in_channels == 9:
        if in_channels == 3 or in_channels == 6:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            # print(in_channels)
            # print(out_channels)
            # dd
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model0(nn.Module):
    def __init__(self,ActionLength=60, num_class=10,base_channel = 64, num_point=24, num_person=1, graph=None, in_channels=6,
                 drop_out=0, adaptive=True):
        super(Model0, self).__init__()

        # if graph is None:
        #     raise ValueError()
        # else:
            # Graph = import_class(graph)
        self.graph = Graph()

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)


        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        # print(x.size())#2x24x6x60

        # N, C, T, V, M = x.size()
        # print(N, C, T, V, M)
        x = x.view(x.size(0),self.num_point,-1, x.size(-1)).permute(0, 2, 3, 1).contiguous().unsqueeze(-1)
        # dd
        # x = fea_npy.permute(0, 3, 1, 2).contiguous()
        # x = x.view(-1, 60, 24 * 6)

        # if len(x.shape) == 3:
        #     N, T, VC = x.shape
        #     x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        # print(N, C, T, V, M)
        # dd

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)
class GCN(nn.Module):
    def __init__(self,ActionLength=60, num_class=14,base_channel = 64, num_point=24, num_person=1, graph=None,
                 in_channels=6,
                 drop_out=0, adaptive=True):
        super(GCN, self).__init__()

        # if graph is None:
        #     raise ValueError()
        # else:
            # Graph = import_class(graph)
        self.graph = Graph()

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)


        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        # print(x.size())#2x24x6x60

        # N, C, T, V, M = x.size()
        # print(N, C, T, V, M)
        x = x.view(x.size(0),self.num_point,-1, x.size(-1)).permute(0, 2, 3, 1).contiguous().unsqueeze(-1)
        # dd
        # print(x.size())
        # dd
        # x = fea_npy.permute(0, 3, 1, 2).contiguous()
        # x = x.view(-1, 60, 24 * 6)

        # if len(x.shape) == 3:
        #     N, T, VC = x.shape
        #     x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        # print(N, C, T, V, M)
        # dd

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # print(x.size())
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)
class Net(nn.Module):
    def __init__(self, ActionLength=60,modelA=ViT(),modelB=LSTM3V1(),modelC=GCN()):
        super(Net, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        # self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        x1 = self.modelA(x)
        # print(x1.size())
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        # print(x2.size())
        x1=F.softmax(x1,dim=-1)
        x2=F.softmax(x2,dim=-1)
        x3=F.softmax(x3,dim=-1)
        x = (x1+x2+x3)/3
        x_1 = x[:,:10]
        #dummpy = -1*torch.ones(x.size(0), 10)
        x_2 = x[:,10:]
        x_2,_ =x_2.max(-1)
        x_2 = x_2.view(-1,1)
        x = torch.cat((x_1, x_2), dim=-1)
        
        # x = (x1+x2)/2
        # x=x3
        # dummpy = torch.ones(x.size(0), 1)#.cuda()
        # x = torch.cat((x, dummpy), dim=-1)
        # x = torch.cat((x1, x2), dim=1)
        # x = self.classifier(F.relu(x))
        return x
