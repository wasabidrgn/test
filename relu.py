import pandas as pd
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
from sklearn import model_selection
import torch
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim, device
import torch.nn.functional as F

# %matplotlib inline
# %config InlineBackend.figure_format='retina'


sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 6
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)



df1 = pd.read_csv("C:\\Users\\wasab\\Desktop\\方案二\\data\\data.csv",encoding='utf-8')
# ,'连接词判断'
df1 = df1.drop(['clause','连接词','Unnamed: 7'],axis=1)
# ,'Unnamed: 6'
pd.set_option("display.width",200)
pd.set_option('display.max_columns',15)
pd.set_option('display.max_colwidth',10)

# print(df1)

df1.loc[df1.结构=='前no','结构'] = 0
df1.loc[df1.结构=='后no','结构'] = 0
df1.loc[df1.结构=='前no后no','结构'] = 0
df1.loc[df1.结构.isnull(),'结构'] = 1


df1.loc[df1.主语改变=='yes','主语改变'] = 1
df1.loc[df1.主语改变=='no','主语改变'] = 0
df1.loc[df1.主语改变.isnull(),'主语改变'] = 2

df1.loc[df1.共享=='yes','共享'] = 1
df1.loc[df1.共享.isnull(),'共享'] = 0

df1.loc[df1.连接词判断=='1','连接词判断'] = 1
df1.loc[df1.连接词判断=='0','连接词判断'] = 0
df1.loc[df1.连接词判断.isnull(),'连接词判断'] = 2

df1.loc[df1.判断结果=='断','判断结果'] = 1
df1.loc[df1.判断结果=='不断','判断结果'] = 0


df1 = df1.astype("float")
print(df1.info)
print(df1.corr(numeric_only = True))

sns.countplot(df1.判断结果)
# plt.show()
# print(df1.判断结果.value_counts() / df1.shape[0])






X = df1[['结构', '主语改变','连接词判断','共享']]
y = df1[['判断结果']]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

X_train = torch.from_numpy(X_train.to_numpy()).float()
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())

X_test = torch.from_numpy(X_test.to_numpy()).float()
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)


class Net(nn.Module):

    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x =
        # return F.relu(self.fc3(x))
        return torch.sigmoid(self.fc3(x))

net = Net(X_train.shape[1])

print(net)

# ax = plt.gca()
# plt.plot(
#   np.linspace(-1, 1, 5),
#   F.relu(torch.linspace(-1, 1, steps=5)).numpy()
# )
# ax.set_ylim([-1.5, 1.5]);
# plt.show()


criterion = nn.BCELoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)

net = net.to(device)
criterion = criterion.to(device)

def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)

def round_tensor(t, decimal_places=4):
    return round(t.item(), decimal_places)


for epoch in range(1000):
    y_pred = net(X_train)
    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)

    if epoch % 100 == 0:
        train_acc = calculate_accuracy(y_train, y_pred)

        y_test_pred = net(X_test)
        y_test_pred = torch.squeeze(y_test_pred)

        test_loss = criterion(y_test_pred, y_test)
        test_acc = calculate_accuracy(y_test, y_test_pred)
        print(f'''epoch {epoch}
              Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
              Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
              ''')

    optimizer.zero_grad()  # 清零梯度缓存
    train_loss.backward()  # 反向传播误差
    optimizer.step()  # 更新参数


MODEL_PATH = 'model.pth'  # 后缀名为 .pth
torch.save(net, MODEL_PATH) # 直接使用torch.save()函数即可

classes = ['不断', '断']

y_pred = net(X_test)
y_pred = y_pred.ge(.5).view(-1).cpu()
y_test = y_test.cpu()

print(classification_report(y_test, y_pred,
                            target_names=classes))

# for i in y_pred:
#     print(i)
# for i in y_test:
#     print(i)