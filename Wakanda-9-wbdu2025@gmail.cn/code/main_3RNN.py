from models.RNN import LSTM3V1
from customize_service import SelectNetworkInput#1 as SelectNetworkInput
import glob
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
os.environ['CUDA_VISIBLE_DEVICES']= '0'



# Training settings
# batch_size = 60
# lr = 3e-4
# gamma = 0.7
# seed = 42
# step_size=5
batch_size = 64
lr = 3e-4
gamma = 0.7
seed = 42
step_size = 4

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cuda'
os.makedirs('data', exist_ok=True)

train_dir = 'train'
test_dir = 'test'

npy_train_path ='/data/duwb/train_new_UP_forward_npy'
npy_train_path1 ='/data/duwb/val'
npy_train_path2 ='/data/duwb/train_new_flip_forward_npy'
npy_train_path3 ='/data/duwb/train_new_UP11_forward_npy'
train_list = glob.glob(npy_train_path+'/*npy')
train_list1 = glob.glob(npy_train_path1+'/*npy')
train_list2 = glob.glob(npy_train_path2+'/*npy')
train_list3 = glob.glob(npy_train_path3+'/*npy')
train_list.extend(train_list1)
train_list.extend(train_list2)
train_list.extend(train_list3)

npy_path1 ='/data/duwb/train_new_forward_npy'
npy_path ='/data/duwb/train_forward_npy'
test_list = glob.glob(npy_path+'/*npy')
test_list1 = glob.glob(npy_path1+'/*npy')
print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")
print(f"Test Data: {len(test_list1)}")
label_match ='/data/duwb/uestc_forward_npy/label_match.npy'
label_match = np.load(label_match,allow_pickle=True)
label_match = label_match.item()
# {2: 0, 5: 1, 6: 2, 7: 3, 13: 4, 14: 5, 17: 6, 19: 7, 20: 8, 22: 9}
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

class CatsDogsDataset(Dataset):
    def __init__(self, file_list,name_style_demo=True, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.name_style_demo = name_style_demo
        self.ActionLength = 60
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        npy_path = self.file_list[idx]
        npy = np.load(npy_path)
        while npy.shape[0] < 120:
            npy = np.concatenate((npy, npy), axis=0)
        npy_random_select =SelectNetworkInput(npy,self.ActionLength)
        npy_transformed = npy_random_select #self.transform(npy_random_select)

        label = npy_path.split("/")[-1].split(".")[0] #a02_c03_forward -->02 is label
        if label[12:15]=='Cam':
            label = int(label[5:7])
            assert label!=10
            if label > 10:
                label = label - 1  # additional class starts from 11 to 14
        elif label[12:15] == 'ard': #forward
            # print(label[8:15])
            # ddd
            label = int(label[1:3])
        # elif self.name_style_demo == 'uestc':
        elif label[-13:-8] == 'color':
            label = label.split("_")[0][1:]  # a9_d8_p040_c1_color_forward
            label_40 = int(label)
            # print(label_40)
            # dd
            if label_40 in label_match.keys():
                label = label_match[label_40]
            else:
                label = 10
        else:
            # print(label[12:15])
            label = int(label[5:7]) - 1  # a02_c03_l_forward -->c03 is label and it starts from 01, so the label is 3-1=2

        return npy_transformed, label

train_data = CatsDogsDataset(train_list,name_style_demo=False, transform=train_transforms)
valid_data = CatsDogsDataset(test_list,name_style_demo=True, transform=test_transforms)
valid_data1 = CatsDogsDataset(test_list1,name_style_demo=True, transform=test_transforms)


train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
valid_loader1 = DataLoader(dataset = valid_data1, batch_size=batch_size, shuffle=True)
ActionLength=60
print(len(train_data), len(train_loader))
print(len(valid_data), len(valid_loader))

model = LSTM3V1(ActionLength=60,hidden_dim1=1024,hidden_dim2=1024,dim_fc=24*6,num_classes=14).to(device)


# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
# save_path ='/data/duwb/models/vit_transformer11.pth'
save_path ='/data/duwb/models_final/Rnn3_Org.pth'
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
def val():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(torch.load(save_path, map_location="cuda:0"))
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(save_path, map_location=device))
        # CPU或者GPU映射
    model.to(device)
    # 声明为推理模式
    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)
    print(epoch_val_accuracy)

def train(epochs):
    acc_best =0
    loss_best =100
    acc_best_sum =0
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for batch_idx,(data,label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            # print(output.size())
            # dd
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx , len(train_loader),
            #            100. * batch_idx / len(train_loader),
            #            loss.item() / len(data)))
        if epoch%3==0:
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in valid_loader:
                    data = data.to(device)
                    label = label.to(device)

                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(valid_loader)
                    epoch_val_loss += val_loss / len(valid_loader)
            with torch.no_grad():
                epoch_val_accuracy1 = 0
                epoch_val_loss1 = 0
                for data, label in valid_loader1:
                    data = data.to(device)
                    label = label.to(device)

                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy1 += acc / len(valid_loader1)
                    epoch_val_loss1 += val_loss / len(valid_loader1)
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - "
                f"val_loss Org/NewRot : {epoch_val_loss:.4f}/{epoch_val_loss1:.4f} "
                f"- val_acc Org/NewRot: {epoch_val_accuracy:.4f}/{epoch_val_accuracy1:.4f}"
                # f"val_loss : {epoch_val_loss1:.4f} - val_acc: {epoch_val_accuracy1:.4f}"
            )
            if epoch_val_accuracy+1.5*epoch_val_accuracy1>=acc_best:
                torch.save(model.state_dict(),save_path)
                acc_best = epoch_val_accuracy+1.5*epoch_val_accuracy1
            if epoch_val_loss+1.5*epoch_val_loss<=loss_best:
                torch.save(model.state_dict(), save_path[:-4] + '_loss.pth')
                loss_best = epoch_val_loss+1.5*epoch_val_loss
            # if epoch_val_accuracy + epoch_accuracy >= acc_best_sum:
            #     torch.save(model.state_dict(), save_path[:-4] + '_all.pth')
            #     acc_best_sum = epoch_val_accuracy + epoch_accuracy

        # if epoch_val_accuracy==1:
        #     torch.save(model.state_dict(), save_path[:-4]+'_e'+str(epoch)+'.pth')
if __name__ == '__main__':
    epochs = 1000
    train(epochs)
    # val()
