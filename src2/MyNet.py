from ctypes import sizeof
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyNet(nn.Module):
    # def __init__(self):
    #     super(MyNet, self).__init__()
    #     self.model = nn.Sequential(
    #         nn.Conv2d(3, 32, 5, 1, 2),
    #         nn.MaxPool2d(2),
    #         nn.Conv2d(32, 32, 5, 1, 2),
    #         nn.MaxPool2d(2),
    #         nn.Conv2d(32, 64, 5, 1, 2),
    #         nn.MaxPool2d(2),
    #         nn.Flatten(),
    #         nn.Linear(1024, 64),
    #         nn.Linear(64, 10),
    #     )


    def __init__(self):
        super(MyNet,self).__init__()
        ########################################################################
        #这里需要写MyNet的卷积层、池化层和全连接层
        # Model 2
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, 9),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(800, 108),
            nn.ReLU(),
            nn.Linear(108, 72),
            nn.ReLU(),
            nn.Linear(72, 10),
            nn.ReLU(),
        )
       
    def forward(self, x):
        ########################################################################
        #这里需要写MyNet的前向传播
        x = self.model(x)
        return x 


def train(net,train_loader,optimizer,n_epochs,loss_function):
    net.train()
    for epoch in range(n_epochs):
        for step, (inputs, labels) in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            ########################################################################
            #计算loss并进行反向传播
            optimizer.zero_grad()
            y_pred = net(inputs)


            # print(y_pred)
            # print(labels)
            # print(y_pred.size())
            # print(labels.size())

            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()
            ########################################################################

            if step % 100 ==0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, step * len(inputs), len(train_loader.dataset), loss.item()))

    print('Finished Training')
    save_path = './MyNet.pth'
    torch.save(net.state_dict(), save_path)

def test(net, test_loader, loss_function): 
    net.eval()
    test_loss = 0.
    num_correct = 0 #correct的个数
    num_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
        ########################################################################
        #需要计算测试集的loss和accuracy
            y_pred = net(inputs)
            pred = y_pred.argmax(dim=1)
            num_correct += (pred == labels).sum()

        accuracy = int(num_correct)/len(test_loader.dataset)
        test_loss = loss_function(y_pred, labels.to(device, torch.long))
           
        ########################################################################
        
        print("Test set: Average loss: {:.4f}\t Acc {:.4f}".format(test_loss.item(), accuracy))
    


def main():
    n_epochs = 5
    train_batch_size = 128
    test_batch_size =5000 
    learning_rate = 5e-4

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)                                      
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,
                                             shuffle=False, num_workers=0)


    net = MyNet()


    # 自己设定优化器和损失函数
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate)
    #######################################################################

    train(net,train_loader,optimizer,n_epochs,loss_function)
    #net.load_state_dict(torch.load('./MyNet_5epochs.pth'))
    test(net,test_loader,loss_function)

if __name__ == '__main__':
   main()
   

    
