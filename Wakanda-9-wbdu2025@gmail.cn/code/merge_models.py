from models.Ensemble4 import ViT
import os
import torch
from models.Ensemble4 import LSTM3V1
from models.Ensemble4 import GCN
from models.Ensemble4 import Net
os.environ['CUDA_VISIBLE_DEVICES']= '3'
device = 'cuda'
# ActionLength=60
print('load model')
PATH1= '/data/duwb/models_final/Vit_Org.pth'
PATH2 ='/data/duwb/models_final/Rnn3_Org.pth'
PATH3 ='/data/duwb/models_final/GCN_Org.pth'
modelA = ViT(ActionLength=60,
    pose_dim=24 * 6,
    num_classes = 14,
    # num_classes = 10,
    dim = 1024,
    # depth = 6,
    depth = 12,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1).to(device)

# Create models and load state_dicts
modelB = LSTM3V1(ActionLength=60,hidden_dim1=1024,hidden_dim2=1024,dim_fc=24*6,num_classes=14).to(device)
modelC = GCN(ActionLength=60,in_channels=6,num_class=14).to(device)
modelA.load_state_dict(torch.load(PATH1))
modelB.load_state_dict(torch.load(PATH2))
modelC.load_state_dict(torch.load(PATH3))

save_path = '/data/duwb/models/vit_rnn_gcn.pth'

model = Net(ActionLength=60,modelA=modelA,modelB=modelB,
            modelC=modelC).to(device)
x= torch.randn(2, 24,6,60).to(device)
output = model(x)
torch.save(model.state_dict(),save_path)