import torch
import numpy as np
import torch.nn as nn
import argparse
from Model import TINet
from torch.nn.init import xavier_uniform_
import torch.utils.data as Data
import matplotlib.pylab as plt
import csv
import codecs


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")


# model initialization  参数初始化
def weight_init(m):
    class_name = m.__class__.__name__  # 得到网络层的名字
    if class_name.find('Conv') != -1:  # 使用了find函数，如果不存在返回值为-1，所以让其不等于-1
        xavier_uniform_(m.weight.data)
    if class_name.find('Linear') != -1:
        xavier_uniform_(m.weight.data)
    if class_name.find('BatchNorm') != -1:
        m.reset_running_stats()
    # if class_name.find('BatchNorm') != -1:
    #     nn.init.constant_(m.weight, 1)
    #     nn.init.constant_(m.bias, 0)


# BN layer initialization


# split train and split data
def data_split_train(data_set, label_set):
    n = int((data_set.shape[0]) * 0.8)
    data_set_train = data_set[:n, :]
    print('data_set_train', data_set_train.shape)
    data_set_val = data_set[n:, :]
    print('data_set_val', data_set_val.shape)
    label_set_train = label_set[:n, :]
    print('label_set_train', label_set_train.shape)
    label_set_val = label_set[n:, :]
    print('label_set_val', label_set_val.shape)
    data_set_train = np.array(data_set_train)
    data_set_val = np.array(data_set_val)
    label_set_train = np.array(label_set_train)
    label_set_val = np.array(label_set_val)

    # data_set_train = []
    # data_set_val = []
    # label_set_train = []
    # label_set_val = []
    # for i in range(data_set.shape[0]):  #行数   shape[2]通道数
    #     index = np.arange(data_set.shape[1])  #列数矩阵[0 1 2 ''']
    #     np.random.shuffle(index)  #随机打乱数据 每次shuffle后数据都被打乱，这个方法可以在机器学习训练的时候在每个epoch结束后将数据重新洗牌进入下一个epoch的学习
    #     a = index[:int((data_set.shape[1]) * 0.8)]
    #     data = data_set[i]  #第i行
    #
    #     data_train = data[a]
    #     data_val = np.delete(data, a, 0)
    #     data_set_train.append(data_train)
    #     data_set_val.append(data_val)
    #     label_set_train.extend(label_set[i][:len(data_train)])
    #     label_set_val.extend(label_set[i][:len(data_val)])
    # data_set_train = np.array(data_set_train).reshape(-1, data_set.shape[-1])
    # data_set_val = np.array(data_set_val).reshape(-1, data_set.shape[-1])
    # label_set_train = np.array(label_set_train)
    # label_set_val = np.array(label_set_val)

    return data_set_train, data_set_val, label_set_train, label_set_val


# training process
def train(train_dataset, val_dataset, batchsize):
    torch.cuda.empty_cache()

    length = len(train_dataset.tensors[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dataloader = Data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_dataloader = Data.DataLoader(val_dataset, batch_size=int(batchsize / 2), shuffle=False)

    val_loss = []
    for epoch in range(args.epochs):
        model.train()
        for index, (data, label) in enumerate(train_dataloader):
            data = data.float().to(device).unsqueeze(dim=1)
            label = label.long().to(device)
            output = model(data).squeeze(-1)  ##########
            # print('output', output.shape)
            loss = criterion(output, label)  # 交叉熵函数 用来判定实际的输出与期望的输出的接近程度 实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(output.data, 1)  # 返回指定维度最大值的序号 dim=1
            correct = pred.eq(label).cpu().sum()
            acc = 100. * correct.item() / len(data)
            if index % 2 == 0:  # 假
                print('Train Epoch: {}/{} [{}/{}] \t Loss: {:.6f} Acc: {:.6f}%'.
                      format(epoch, args.epochs, [(index + 1) * batchsize if len(data) == batchsize else length][0],
                             length, loss.item(), acc))  # batchsize=100

        # validation
        model.eval()
        correct_val = 0
        sum_loss = 0
        length_val = len(val_dataset)
        for index, (data_val, label_val) in enumerate(val_dataloader):
            with torch.no_grad():
                data_val = data_val.float().to(device).unsqueeze(dim=1)
                label_val = label_val.long().to(device)
                output_val = model(data_val)
                loss = criterion(output_val, label_val)

                pred_val = torch.argmax(output_val.data, 1)
                correct_val += pred_val.eq(label_val).cpu().sum()
                sum_loss += loss
        acc = 100. * correct_val.item() / length_val
        average_loss = sum_loss.item() / length_val

        print('\n The {}/{} epoch result : Average loss: {:.6f}, Acc_val: {:.2f}%'.format(
            epoch, args.epochs, average_loss, acc))
        val_loss.append(loss.item())
        '''可视化操作'''
        # accData.append([epoch,acc])
        # lossData.append([epoch,average_loss])
    # torch.save(model.state_dict(), 'model.txt')
    plt.plot(val_loss)
    plt.show()
    # data_write_csv(".\\loss_1d_LinGang.csv", lossData)
    # data_write_csv(".\\acc_1d_LinGang.csv", accData)


# testing
def tst(test_dataset_s):
    model.eval()
    length = len(test_dataset_s)
    correct = 0
    test_loader = Data.DataLoader(test_dataset_s, batch_size=100, shuffle=False)
    for index, (data, label) in enumerate(test_loader):
        with torch.no_grad():
            data = data.float().to(device)
            label = label.long().to(device)

            output = model(data.unsqueeze(dim=1)).squeeze(-1)  ###########
            pred = torch.argmax(output.data, 1)
            correct += pred.eq(label).cpu().sum()

    acc = 100. * correct / length

    return acc


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # use cpu or gpu
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # load data
    dataset_s_train = np.load(r'data\processed\1024.npy', allow_pickle=True)
    label_s_train = np.load(r'data\processed\1024labels.npy', allow_pickle=True)
    # dataset_s_train = dataset_s_train.transpose((1, 0))  # 转置
    data1 = []
    for i in range(120):
        a = label_s_train[i, :]
        if a == 'normal':
            a = 0
        elif a == 'inner':
            a = 1
        elif a == 'outer':
            a = 2
        else:
            a = 3
        data1.append(a)
    # data0 = pd.read_csv(r'E:\competition\AnalysisData\label_train.csv', usecols=[1])
    data1 = np.array(data1).reshape(120, 1)
    print('data1', data1.shape)
    label_s_train = data1

    # 划分训练集和测试集
    data_s_train_val = dataset_s_train[:95, :]
    label_s_train_val = label_s_train[:95, :]
    data_s_test = dataset_s_train[95:, :]
    label_s_test = label_s_train[95:, :]
    print('data_s_train_val', data_s_train_val.shape)
    print('label_s_train_val', label_s_train_val.shape)
    print('data_s_test', data_s_test.shape)
    print('label_s_test', label_s_test.shape)

    iteration_acc = []

    # data_s_test = data_s_test.reshape(-1, 1024)
    # label_s_test = label_s_test.reshape(1, -1)   #转化成一行

    # # set hyper-parameters 创建 ArgumentParser() 对象
    parser = argparse.ArgumentParser(description='standard training')  # 在参数帮助文档之前显示的文本
    # 调用 add_argument() 方法添加参数
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--iterations', type=int, default=1, help='number of iteration')
    parser.add_argument('--batch_size', type=int, default=100, help='training batch_size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    args = parser.parse_args()  # 解析添加的参数
    # repeat several times for an average result
    for iteration in range(args.iterations):
        # load model
        model = TINet(C_in=1, class_num=4).to(device)  #
        model.apply(weight_init)
        # 划分训练集和验证集
        data_s_train, data_s_val, label_s_train, label_s_val = data_split_train(data_s_train_val, label_s_train_val)
        # transfer ndarray to tensor
        torch_data_train = torch.from_numpy(data_s_train)
        torch_data_val = torch.from_numpy(data_s_val)
        torch_data_test = torch.from_numpy(data_s_test)
        torch_label_train = torch.from_numpy(label_s_train)
        torch_label_val = torch.from_numpy(label_s_val)
        torch_label_test = torch.from_numpy(label_s_test)
        # seal to data-set
        train_dataset_s = Data.TensorDataset(torch_data_train, torch_label_train)
        val_dataset_s = Data.TensorDataset(torch_data_val, torch_label_val)
        test_dataset_s = Data.TensorDataset(torch_data_test, torch_label_test.squeeze())

        criterion = nn.NLLLoss()  # 损失函数
        train(train_dataset_s, val_dataset_s, args.batch_size)
        acc = tst(test_dataset_s)
        iteration_acc.append(acc)

        print("test_accuracy: %.6f" % acc)

    print("Average Testing Accuracy in " + str(args.iterations) + " iterations : %.5f" %
          np.array(iteration_acc).mean())
