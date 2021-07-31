from time import time
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.utils.data import TensorDataset, DataLoader
from datapreprocessing import data_train_fft, code, scaler, encoder
from Model import TINet

learning_rate = 0.001
batch_size = 64
epochs = 30

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

data = torch.from_numpy(data_train_fft)
code = torch.from_numpy(code)

data_train = data[:95, :]
label_train = code[:95]
data_test = data[95:, :]
label_test = code[95:]

train_dataset = TensorDataset(data_train, label_train)
test_dataset = TensorDataset(data_test, label_test)


def weight_init(m):
    class_name = m.__class__.__name__  # 得到网络层的名字
    if class_name.find('Conv') != -1:  # 使用了find函数，如果不存在返回值为-1，所以让其不等于-1
        xavier_uniform_(m.weight.data)
    if class_name.find('Linear') != -1:
        xavier_uniform_(m.weight.data)
    if class_name.find('BatchNorm') != -1:
        m.reset_running_stats()


model = TINet(C_in=1, class_num=4).to(device)
model.apply(weight_init)

criterion = nn.NLLLoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

loss_list = []

print('Training Starts!')
start = time()
for epoch in range(epochs):
    model.train()
    for index, (data, label) in enumerate(train_dataloader):
        data = data.float().to(device).unsqueeze(dim=1)
        label = label.long().to(device)

        output = model(data).squeeze(-1)
        loss = criterion(output, label)  # 交叉熵函数 用来判定实际的输出与期望的输出的接近程度 实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = output.argmax(axis=1).eq(label).sum() / batch_size

        if index == 0:
            print(f'Epoch: {epoch}\tLoss: {loss.item():.3f}\tAcc: {correct:.3f}')
            loss_list.append(loss.item())

print(f'Finished! Time used: {time() - start:.3f}s')

model.eval()
correct = 0
for index, (data, label) in enumerate(test_dataloader):
    data = data.float().to(device).unsqueeze(dim=1)
    label = label.long().to(device)
    output = model(data).squeeze(-1)
    correct += output.argmax(axis=1).eq(label).sum()
accuracy = correct / 25
print(f'Accuracy on test set: {accuracy:.3f}')

exit()
