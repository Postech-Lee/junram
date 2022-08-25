from torch.utils.data import DataLoader # train,test 데이터를 loader객체로 만들어주기 위해서
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    )

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')


class DMDM(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(64,)  # 합성곱 연산 (입력 채널 수: 3, 출력 채널 수: 6, 필터 크기: 5x5, stride=1(default))
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)  # 합성곱 연산 (필터크기 2x2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)  # 합성곱 연산 (입력 채널 수: 6, 출력 채널수: 16, 필터 크기: 5x5, stride=1(default))
        self.pool2 = nn.MaxPool2d(2, 2)  # 합성곱 연산 (필터크기 2x2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 피쳐맵 16개를 일렬로 피면 16*5*5개의 노드가 생성됨.
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # conv1 -> ReLU -> pool1
        x = self.pool2(F.relu(self.conv2(x)))  # conv2 -> ReLU -> pool2

        x = x.view(-1, 16 * 5 * 5)  # 5x5 피쳐맵 16개를 일렬로 만든다.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


net = Net().to(device)  # 모델 선언
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

loss_ = []  # loss 저장용 리스트
n = len(trainloader)  # 배치개수

for epoch in range(3):  # 10회 반복

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)  # 배치 데이터

        optimizer.zero_grad()  # 배치마다 optimizer 초기화

        outputs = net(inputs)  # 노드 10개짜리 예측값 산출
        loss = criterion(outputs, labels)  # 크로스 엔트로피 손실함수 계산    optimizer.zero_grad() # 배치마다 optimizer 초기화
        loss.backward()  # 손실함수 기준 역전파
        optimizer.step()  # 가중치 최적화

        running_loss += loss.item()

    loss_.append(running_loss / n)
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

PATH = './cifar_net.pth' # 모델 저장 경로
torch.save(net.state_dict(), PATH) # 모델 저장장
net = Net().to(device) # 모델 선언
net.load_state_dict(torch.load(PATH)) # 모델 parameter 불러오기
correct = 0
total = 0
with torch.no_grad(): # 파라미터 업데이트 같은거 안하기 때문에 no_grad를 사용.
  # net.eval() # batch normalization이나 dropout을 사용하지 않았기 때문에 사용하지 않음. 항상 주의해야함.
  for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1) # 10개의 class중 가장 값이 높은 것을 예측 label로 추출.
    total += labels.size(0) # test 개수
    correct += (predicted == labels).sum().item() # 예측값과 실제값이 맞으면 1 아니면 0으로 합산.

print(f'accuracy of 10000 test images: {100*correct/total}%')
print(outputs.data)
print(predicted)