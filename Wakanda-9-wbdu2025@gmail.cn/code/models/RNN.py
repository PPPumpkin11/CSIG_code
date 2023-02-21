import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
class HRNNV2Est3C1(nn.Module):
    def __init__(self,seq_len,hidden_dim1,hidden_dim2,dim_fc=24*6,mean_dim=1,
                 input_channel=3,num_classes=10, aux_logits=True, transform_input=False):
        super(HRNNV2Est3C1, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        ##########att setting##########
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)
        self.mean_dim=mean_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.seq_len = seq_len
        self.use_gpu = torch.cuda.is_available()
        self.dim_fc = dim_fc
        self.lstm_input_dim = self.dim_fc
        self.sign = LBSign.apply
        self.lstm_7a = nn.LSTM(self.dim_fc, hidden_dim1,batch_first=True)
        self.lstm_7b = nn.LSTM(hidden_dim1, hidden_dim2,batch_first=True)
        self.classifier = nn.Linear(hidden_dim2, num_classes)
        self.linear_att_RL=self._make_layers_att_RL(self.dim_fc+hidden_dim1+hidden_dim2,)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def init_hidden(self,hidden_dim,seq_num=None):
        if seq_num is None:
            seq_num= self.seq_num*self.num_att
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, seq_num, hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, seq_num, hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, seq_num, hidden_dim))
            c0 = Variable(torch.zeros(1,seq_num, hidden_dim))
        return (h0, c0)
    def _make_layers_pose_att(self, cfg,in_channels):
        layers = []
        x0_dim=cfg[0]
        x1_dim=cfg[1] #joint heatmap number
        layers += [nn.Conv1d(in_channels, x0_dim, kernel_size=1, padding=0),
                   # nn.BatchNorm2d(x),
                   nn.Tanh(),
                   nn.Conv1d(x0_dim, x1_dim, kernel_size=1, padding=0)]
            # in_channels = x
        return nn.Sequential(*layers)#, nn.Sequential(*layers_1)
    def _make_layers_att(self, cfg):
        layers = []
        x0_dim=cfg[0]
        x1_dim=cfg[1] #joint heatmap number
        layers += [nn.Tanh(),
                   nn.Conv1d(x0_dim, x1_dim, kernel_size=1, padding=0)]
            # in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)#, nn.Sequential(*layers_1)
    def _make_layers_att_RL(self, cfg):
        layers = []
        layers += [nn.Linear(cfg, 1024),
                   nn.ReLU(),
                  nn.Linear(1024,256),
                  nn.ReLU(),
                  nn.Linear(256,3)]
        return nn.Sequential(*layers)#, nn.Sequential(*layers_1)
    def forward(self, x0,param_a=1):
        # 299 x 299 x 3
        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()

        vid_num =x0.size(0)
        # print(vid_num)
        vid_len =x0.size(1)
        #################
        conv_out_7a_sep = x0#out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        probs_sup =[]
        #get the baseline for RL:
        # hidden_init= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim)
        hidden_a= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1)
        hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        for i in range(0,vid_len):
            lstm_input_i=conv_out_7a_sep[:,i,:]
            lstm_input_i=lstm_input_i.contiguous().view(vid_num,1,-1)
            hidden_tmp1 = hidden_a[0].view(-1,hidden_a[0].size(-1))
            hidden_tmp2 = hidden_b[0].view(-1,hidden_b[0].size(-1))
            lstm_input_i_= lstm_input_i.view(lstm_input_i.size(0),-1)
            hidden_tmp = torch.cat([lstm_input_i_,hidden_tmp1,hidden_tmp2],dim=-1) #TODO modify the function
            hidden_proj_tmp_ = self.linear_att_RL(hidden_tmp)
            if self.training:
                probs = F.gumbel_softmax(hidden_proj_tmp_,tau=1/param_a,hard=True)
                _, alpha = probs.max(-1)
                alpha = alpha.float()
                alpha0 = probs[:, 0]  # do not split
                alpha1 = probs[:, 1]  # org alpha: split-->split and keep
                alpha2 = probs[:, 2]  # split and abandon
            else:
                _, alpha = hidden_proj_tmp_.max(-1)
                alpha=alpha.float()
                alpha0=  alpha==0
                alpha1=  alpha==1
                alpha2=  alpha==2
                alpha0 = alpha0.float()
                alpha1 = alpha1.float()
                alpha2 = alpha2.float()
            alpha0 = alpha0.view(1, -1)
            alpha1 = alpha1.view(1, -1)
            alpha2 = alpha2.view(1, -1)
            hidden_proj_tmp_0 =hidden_proj_tmp_.view(hidden_proj_tmp_.size(0),1,-1)
            probs_sup.append(hidden_proj_tmp_0)#32x1
            lstm_out_a_, hidden_a_ = self.lstm_7a(lstm_input_i, hidden_a)
            lstm_out_b_, hidden_b_ = self.lstm_7b(lstm_out_a_, hidden_b)
            #update 1layer h,c
            h_update_a=alpha1[:,:,None]*hidden_a_[0]+alpha2[:,:,None]*hidden_a[0]
            c_update_a = alpha1[:, :, None] * hidden_a_[1] + alpha2[:, :, None] * hidden_a[1]
            hidden_a=[h_update_a,c_update_a]
            #update 2layer h,c

            h_update_b=alpha0[:,:,None]*hidden_b_[0]+alpha1[:,:,None]*hidden_b[0]\
                       +alpha2[:,:,None]*hidden_b[0]

            c_update_b = alpha0[:, :, None] * hidden_b_[1] + alpha1[:, :, None] * hidden_b[1] \
                         + alpha2[:, :, None] * hidden_b[1]
            action_all_base.append(alpha1.view(-1, 1))
            hidden_b=[h_update_b,c_update_b]
        lstm_out_base = lstm_out_b_.sum(1)#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        action_all_base = torch.cat(action_all_base, dim=-1)
        probs_sup = torch.cat(probs_sup, dim=-1)
        return x_out_base,action_all_base#,probs_sup#,hidden_att
class LSTMV1(nn.Module):
    def __init__(self,seq_len,hidden_dim1,hidden_dim2,dim_fc=24*6,mean_dim=1,
                 input_channel=3,num_classes=10, aux_logits=True, transform_input=False):
        super(LSTMV1, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        ##########att setting##########
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.seq_len = seq_len
        self.use_gpu = torch.cuda.is_available()
        self.dim_fc = dim_fc
        self.lstm_input_dim = self.dim_fc
        self.sign = LBSign.apply
        self.lstm_7a = nn.LSTM(self.dim_fc, hidden_dim1,batch_first=True)
        # self.lstm_7b = nn.LSTM(hidden_dim1, hidden_dim2,batch_first=True)
        self.classifier = nn.Linear(hidden_dim2, num_classes)
        # self.linear_att_RL=self._make_layers_att_RL(self.dim_fc+hidden_dim1+hidden_dim2,)
    def init_hidden(self,hidden_dim,seq_num=None):
        if seq_num is None:
            seq_num= self.seq_num*self.num_att
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, seq_num, hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, seq_num, hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, seq_num, hidden_dim))
            c0 = Variable(torch.zeros(1,seq_num, hidden_dim))
        return (h0, c0)
    def forward(self, x0,param_a=1):
        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()
        vid_num =x0.size(0)
        vid_len =x0.size(1)
        #################
        conv_out_7a_sep = x0#out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        hidden_a= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1)
        # hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        for i in range(0,vid_len):
            lstm_input_i=conv_out_7a_sep[:,i,:]
            lstm_input_i=lstm_input_i.contiguous().view(vid_num,1,-1)
            lstm_out_a_, hidden_a = self.lstm_7a(lstm_input_i, hidden_a)
            # lstm_out_b_, hidden_b_ = self.lstm_7b(lstm_out_a_, hidden_b)
            #update 1layer h,c
        # print(lstm_out_a_.size())
        # dd
        lstm_out_base = lstm_out_a_.sum(1)#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        return x_out_base,action_all_base#,probs_sup#,hidden_att
class LSTMV2(nn.Module):
    def __init__(self,seq_len,hidden_dim1,hidden_dim2,dim_fc=24*6,mean_dim=1,
                 input_channel=3,num_classes=10, aux_logits=True, transform_input=False):
        super(LSTMV2, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        ##########att setting##########
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.seq_len = seq_len
        self.use_gpu = torch.cuda.is_available()
        self.dim_fc = dim_fc
        self.lstm_input_dim = self.dim_fc
        self.sign = LBSign.apply
        self.lstm_7a = nn.LSTM(self.dim_fc, hidden_dim1,batch_first=True)
        self.lstm_7b = nn.LSTM(hidden_dim1, hidden_dim2,batch_first=True)
        self.classifier = nn.Linear(hidden_dim2, num_classes)
        # self.linear_att_RL=self._make_layers_att_RL(self.dim_fc+hidden_dim1+hidden_dim2,)
    def init_hidden(self,hidden_dim,seq_num=None):
        if seq_num is None:
            seq_num= self.seq_num*self.num_att
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, seq_num, hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, seq_num, hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, seq_num, hidden_dim))
            c0 = Variable(torch.zeros(1,seq_num, hidden_dim))
        return (h0, c0)
    def forward(self, x0,param_a=1):
        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()
        vid_num =x0.size(0)
        vid_len =x0.size(1)
        #################
        conv_out_7a_sep = x0#out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        hidden_a= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1)
        hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        for i in range(0,vid_len):
            lstm_input_i=conv_out_7a_sep[:,i,:]
            lstm_input_i=lstm_input_i.contiguous().view(vid_num,1,-1)
            lstm_out_a_, hidden_a = self.lstm_7a(lstm_input_i, hidden_a)
            # print(lstm_out_a_.size())
            # dd
            lstm_out_b_, hidden_b = self.lstm_7b(lstm_out_a_, hidden_b)
            #update 1layer h,c
        # print(lstm_out_a_.size())
        # dd
        lstm_out_base = lstm_out_b_.sum(1)#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        return x_out_base,action_all_base#,probs_sup#,hidden_att
class LSTM2V1(nn.Module):
    def __init__(self,ActionLength=60,hidden_dim1=1024,hidden_dim2=1024,dim_fc=24*6,mean_dim=1,
                 input_channel=3,num_classes=10,drop=0., aux_logits=True, transform_input=False):
        super(LSTM2V1, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        ##########att setting##########
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        # self.seq_len = seq_len
        self.use_gpu = torch.cuda.is_available()
        self.dim_fc = dim_fc
        self.lstm_input_dim = self.dim_fc
        self.sign = LBSign.apply
        self.lstm_7a = nn.LSTM(self.dim_fc, hidden_dim1,num_layers=2,batch_first=True)
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
    def forward(self, x0,param_a=1):
        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()
        vid_num =x0.size(0)
        vid_len =x0.size(1)
        #################
        conv_out_7a_sep = x0#out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        hidden_a= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1,num_layer=2)
        # hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        lstm_out_b_, hidden_b = self.lstm_7a(conv_out_7a_sep, hidden_a)
        lstm_out_base = lstm_out_b_[:,-1,:]#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        return x_out_base,action_all_base#,probs_sup#,hidden_att
class LSTM3V1(nn.Module):
    def __init__(self,ActionLength=60,hidden_dim1=1024,hidden_dim2=1024,dim_fc=24*6,mean_dim=1,
                 input_channel=3,num_classes=10,drop=0., aux_logits=True, transform_input=False):
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
    def forward(self, x0,param_a=1):

        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()
        vid_num =x0.size(0)
        vid_len =x0.size(1)

        x0 = torch.cat([x0,x0],dim=1)
        #################
        conv_out_7a_sep = x0#out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        hidden_a= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1,num_layer=3)
        # hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        lstm_out_b_, hidden_b = self.lstm_7a(conv_out_7a_sep, hidden_a)
        lstm_out_base = lstm_out_b_[:,-1,:]#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        return x_out_base#,action_all_base#,probs_sup#,hidden_att
class LSTMnV1(nn.Module):
    def __init__(self,ActionLength=60,hidden_dim1=1024,hidden_dim2=1024,dim_fc=24*6,num_layers=3,
                 input_channel=3,num_classes=10,drop=0., aux_logits=True, transform_input=False):
        super(LSTMnV1, self).__init__()
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
        self.num_layers = num_layers
        # self.sign = LBSign.apply
        self.lstm_7a = nn.LSTM(self.dim_fc, hidden_dim1,dropout=drop,num_layers=num_layers,batch_first=True)
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
    def forward(self, x0,param_a=1):
        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()
        vid_num =x0.size(0)
        vid_len =x0.size(1)
        #################
        conv_out_7a_sep = x0#out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        hidden_a= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1,num_layer=self.num_layers)
        # hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        lstm_out_b_, hidden_b = self.lstm_7a(conv_out_7a_sep, hidden_a)
        lstm_out_base = lstm_out_b_[:,-1,:]#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        return x_out_base,action_all_base#,probs_sup#,hidden_att
class LSTM3Bi(nn.Module):
    def __init__(self,ActionLength=60,hidden_dim1=1024,hidden_dim2=1024,dim_fc=24*6,mean_dim=1,
                 input_channel=3,num_classes=10,drop=0., aux_logits=True, transform_input=False):
        super(LSTM3Bi, self).__init__()
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
        self.lstm_7a = nn.LSTM(self.dim_fc, hidden_dim1,dropout=drop,bidirectional=True,num_layers=3,batch_first=True)
        # self.lstm_7b = nn.LSTM(hidden_dim1, hidden_dim2,batch_first=True)
        self.classifier = nn.Linear(hidden_dim2*2, num_classes)
        # self.linear_att_RL=self._make_layers_att_RL(self.dim_fc+hidden_dim1+hidden_dim2,)
    def init_hidden(self,hidden_dim,num_layer=1,seq_num=None,bidirectional=False):
        if bidirectional:
            double =2
        else:
            double =1
        if seq_num is None:
            seq_num= self.seq_num*self.num_att
        if self.use_gpu:
            h0 = Variable(torch.zeros(double*num_layer, seq_num, hidden_dim).cuda())
            c0 = Variable(torch.zeros(double*num_layer, seq_num, hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(double*num_layer, seq_num, hidden_dim))
            c0 = Variable(torch.zeros(double*num_layer,seq_num, hidden_dim))
        return (h0, c0)
    def forward(self, x0,param_a=1):
        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()
        vid_num =x0.size(0)
        vid_len =x0.size(1)
        #################
        conv_out_7a_sep = x0#out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        hidden_a= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1,num_layer=3,bidirectional=True)
        # hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        lstm_out_b_, hidden_b = self.lstm_7a(conv_out_7a_sep, hidden_a)
        # print(lstm_out_b_.size())
        # print(hidden_b[0].size())
        # print(len(hidden_b))
        # dd
        lstm_out_base = lstm_out_b_[:,-1,:]#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        return x_out_base,action_all_base#,probs_sup#,hidden_att
class LSTM3Bi1(nn.Module):
    def __init__(self,ActionLength=60,hidden_dim1=1024,hidden_dim2=1024,dim_fc=24*6,mean_dim=1,
                 input_channel=3,num_classes=10,drop=0., aux_logits=True, transform_input=False):
        super(LSTM3Bi1, self).__init__()
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
        self.lstm_7a = nn.LSTM(self.dim_fc, hidden_dim1,dropout=drop,bidirectional=True,num_layers=3,batch_first=True)
        # self.lstm_7b = nn.LSTM(hidden_dim1, hidden_dim2,batch_first=True)
        self.classifier = nn.Linear(hidden_dim2*2, num_classes)
        # self.linear_att_RL=self._make_layers_att_RL(self.dim_fc+hidden_dim1+hidden_dim2,)
    def init_hidden(self,hidden_dim,num_layer=1,seq_num=None,bidirectional=False):
        if bidirectional:
            double =2
        else:
            double =1
        if seq_num is None:
            seq_num= self.seq_num*self.num_att
        if self.use_gpu:
            h0 = Variable(torch.zeros(double*num_layer, seq_num, hidden_dim).cuda())
            c0 = Variable(torch.zeros(double*num_layer, seq_num, hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(double*num_layer, seq_num, hidden_dim))
            c0 = Variable(torch.zeros(double*num_layer,seq_num, hidden_dim))
        return (h0, c0)
    def forward(self, x0,param_a=1):
        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()
        vid_num =x0.size(0)
        vid_len =x0.size(1)
        #################
        conv_out_7a_sep = x0#out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        hidden_a= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1,num_layer=3,bidirectional=True)
        # hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        lstm_out_b_, hidden_b = self.lstm_7a(conv_out_7a_sep, hidden_a)
        # print(lstm_out_b_.size())
        # print(hidden_b[0].size())
        # print(len(hidden_b))
        # dd
        lstm_out_base = lstm_out_b_.mean(1)#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        return x_out_base,action_all_base#,probs_sup#,hidden_att
class LSTM3Bi2(nn.Module):
    def __init__(self,ActionLength=60,hidden_dim1=1024,hidden_dim2=1024,dim_fc=24*6,mean_dim=1,
                 input_channel=3,num_classes=10,drop=0., aux_logits=True, transform_input=False):
        super(LSTM3Bi2, self).__init__()
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
        self.lstm_1 = nn.LSTM(self.dim_fc, hidden_dim1,dropout=drop,num_layers=3,batch_first=True)
        self.lstm_2 = nn.LSTM(self.dim_fc, hidden_dim1,dropout=drop,num_layers=3,batch_first=True)
        # self.lstm_7b = nn.LSTM(hidden_dim1, hidden_dim2,batch_first=True)
        self.classifier = nn.Linear(hidden_dim2*2, num_classes)
        # self.linear_att_RL=self._make_layers_att_RL(self.dim_fc+hidden_dim1+hidden_dim2,)
    def init_hidden(self,hidden_dim,num_layer=1,seq_num=None,bidirectional=False):
        if bidirectional:
            double =2
        else:
            double =1
        if seq_num is None:
            seq_num= self.seq_num*self.num_att
        if self.use_gpu:
            h0 = Variable(torch.zeros(double*num_layer, seq_num, hidden_dim).cuda())
            c0 = Variable(torch.zeros(double*num_layer, seq_num, hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(double*num_layer, seq_num, hidden_dim))
            c0 = Variable(torch.zeros(double*num_layer,seq_num, hidden_dim))
        return (h0, c0)
    def forward(self, x0,param_a=1):
        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()
        vid_num =x0.size(0)
        vid_len =x0.size(1)
        #################
        lstm_forward = x0 #out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        lstm_backward = torch.flip(x0, [1]) #out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        hidden_1= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1,num_layer=3,)
        hidden_2= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1,num_layer=3,)
        # hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        lstm_out_f_, hidden_1 = self.lstm_1(lstm_forward, hidden_1)
        lstm_out_b_, hidden_2 = self.lstm_2(lstm_backward, hidden_2)
        # print(lstm_out_b_.size())
        # print(hidden_b[0].size())
        # print(len(hidden_b))
        # dd
        # print(lstm_out_f_.size())
        lstm_out_base = torch.cat([lstm_out_f_[:,-1,:],lstm_out_b_[:,-1,:]],dim=-1)
        # dd
        # lstm_out_base = lstm_out_b_.mean(1)#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        return x_out_base,action_all_base#,probs_sup#,hidden_att
class GRU3V1(nn.Module):
    def __init__(self,ActionLength=60,hidden_dim1=1024,hidden_dim2=1024,dim_fc=24*6,mean_dim=1,
                 input_channel=3,num_classes=10, aux_logits=True, transform_input=False):
        super(GRU3V1, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        ##########att setting##########
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.seq_len = ActionLength
        self.use_gpu = torch.cuda.is_available()
        self.dim_fc = dim_fc
        self.lstm_input_dim = self.dim_fc
        # self.sign = LBSign.apply
        self.lstm_7a = nn.GRU(self.dim_fc, hidden_dim1,num_layers=3,batch_first=True)
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

    def init_hidden_gru(self, hidden_dim, num_layer=1, seq_num=None):
        if seq_num is None:
            seq_num = self.seq_num * self.num_att
        if self.use_gpu:
            h0 = Variable(torch.zeros(num_layer, seq_num, hidden_dim).cuda())
            # c0 = Variable(torch.zeros(num_layer, seq_num, hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(num_layer, seq_num, hidden_dim))
            # c0 = Variable(torch.zeros(num_layer, seq_num, hidden_dim))
        return h0
    def forward(self, x0,param_a=1):
        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()
        vid_num =x0.size(0)
        vid_len =x0.size(1)
        #################
        conv_out_7a_sep = x0#out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        hidden_a= self.init_hidden_gru(seq_num=vid_num,hidden_dim=self.hidden_dim1,num_layer=3)
        # hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        lstm_out_b_, hidden_b = self.lstm_7a(conv_out_7a_sep, hidden_a)
        lstm_out_base = lstm_out_b_[:,-1,:]#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        return x_out_base,action_all_base#,probs_sup#,hidden_att
class LSTM4V1(nn.Module):
    def __init__(self,ActionLength=60,hidden_dim1=1024,hidden_dim2=1024,dim_fc=24*6,mean_dim=1,
                 input_channel=3,num_classes=10, aux_logits=True, transform_input=False):
        super(LSTM4V1, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        ##########att setting##########
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.seq_len = ActionLength
        self.use_gpu = torch.cuda.is_available()
        self.dim_fc = dim_fc
        self.lstm_input_dim = self.dim_fc
        # self.sign = LBSign.apply
        self.lstm_7a = nn.LSTM(self.dim_fc, hidden_dim1,num_layers=4,batch_first=True)
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
    def forward(self, x0,param_a=1):
        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()
        vid_num =x0.size(0)
        vid_len =x0.size(1)
        #################
        conv_out_7a_sep = x0#out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        hidden_a= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1,num_layer=4)
        # hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        lstm_out_b_, hidden_b = self.lstm_7a(conv_out_7a_sep, hidden_a)
        lstm_out_base = lstm_out_b_[:,-1,:]#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        return x_out_base,action_all_base#,probs_sup#,hidden_att
class LSTMBi(nn.Module):
    def __init__(self,ActionLength=60,hidden_dim1=1024,hidden_dim2=1024,dim_fc=24*6,mean_dim=1,
                 input_channel=3,num_classes=10, aux_logits=True, transform_input=False):
        super(LSTMBi, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        ##########att setting##########
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.seq_len = ActionLength
        self.use_gpu = torch.cuda.is_available()
        self.dim_fc = dim_fc
        self.lstm_input_dim = self.dim_fc
        # self.sign = LBSign.apply
        self.lstm_7a = nn.LSTM(self.dim_fc, hidden_dim1,
                               bidirectional=True,
                               num_layers=1,batch_first=True)
        # self.lstm_7b = nn.LSTM(hidden_dim1, hidden_dim2,batch_first=True)
        self.classifier = nn.Linear(hidden_dim2*2, num_classes)
        # self.linear_att_RL=self._make_layers_att_RL(self.dim_fc+hidden_dim1+hidden_dim2,)
    def init_hidden(self,hidden_dim,num_state=1,seq_num=None):
        if seq_num is None:
            seq_num= self.seq_num*self.num_att
        if self.use_gpu:
            h0 = Variable(torch.zeros(num_state, seq_num, hidden_dim).cuda())
            c0 = Variable(torch.zeros(num_state, seq_num, hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(num_state, seq_num, hidden_dim))
            c0 = Variable(torch.zeros(num_state,seq_num, hidden_dim))
        return (h0, c0)
    def forward(self, x0,param_a=1):
        x0= x0.view(x0.size(0),-1,x0.size(3))
        x0 = x0.permute(0,2,1).contiguous()
        vid_num =x0.size(0)
        vid_len =x0.size(1)
        #################
        conv_out_7a_sep = x0#out.view(vid_num,vid_len,self.dim_fc, -1) # N*512*k
        action_all_base =[]
        hidden_a= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim1,num_state=2)
        # hidden_b= self.init_hidden(seq_num=vid_num,hidden_dim=self.hidden_dim2)
        lstm_out_b_, hidden_b = self.lstm_7a(conv_out_7a_sep, hidden_a)
        # print(lstm_out_b_.size())
        # dd
        lstm_out_base = lstm_out_b_[:,-1,:]#.contiguous()
        x_out_base = self.classifier(lstm_out_base)
        return x_out_base,action_all_base#,probs_sup#,hidden_att
class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,param_a=1):
        # return torch.sign(input)
        ret = 0.5*(param_a*input+1)
        ret = ret.clamp_(0,1)
        # print(ret)
        ret = ret>0.5
        ret = ret.float()
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        # return grad_output.clamp_(-1, 1)
        return grad_output.clamp_(0, 1)