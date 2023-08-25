import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from PIL import Image
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt  # 임포트 추가
from mpl_toolkits.mplot3d import Axes3D  # 임포트 추가
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

image_size=4
def generate_sample_from_image(image_path, index):
    image = Image.open(image_path)
    image = image.resize((image_size, image_size))
    image = np.array(image)
    if index % 2 == 0:
        text = "I was a victim of voice phishing"
    else:
        text = "I was a victim of smishing"
    
    numeric = np.random.rand(12)
    label = index % 2  # 인덱스에 따라 0 또는 1 부여
    
    return {'image': image, 'text': text, 'numeric': numeric, 'y': label}

# 데이터 샘플 생성

# 이미지 파일 경로
image_file_paths = ['./data/phone.png', './data/text.png','./data/phone.png', './data/text.png',
                    './data/phone.png', './data/text.png','./data/phone.png', './data/text.png',
                    './data/phone.png', './data/text.png','./data/phone.png', './data/text.png']

# 데이터 샘플 생성
desired_text_feature_length=30
data_samples_from_images = [generate_sample_from_image(image_path, i) for i, image_path in enumerate(image_file_paths)]
data_list = []
for sample in data_samples_from_images:
    image_feature = sample['image'].flatten()  # 이미지 데이터를 1D 벡터로 변환
    text_feature = [ord(char) for char in sample['text']]
    text_feature_padded = np.pad(text_feature, (0, max(0, desired_text_feature_length - len(text_feature))), mode='constant')
    text_feature_padded = text_feature_padded[:desired_text_feature_length]  # Truncate if needed
    
    numeric_feature = sample['numeric']
    
    y = torch.tensor(sample['y'], dtype=torch.float)

    # 이미지, 텍스트, 숫자 데이터를 PyTorch 텐서로 변환
    x_image = torch.tensor(image_feature, dtype=torch.float).view(-1, 1)
    x_text = torch.tensor(text_feature_padded, dtype=torch.float).view(-1, 1)
    x_numeric = torch.tensor(numeric_feature, dtype=torch.float).view(-1, 1)
    
    # 각각의 데이터 텐서를 연결하여 노드 속성 생성
    x = torch.cat([x_image, x_text, x_numeric], dim=0)
    
    # 노드 간의 엣지 생성
    num_nodes = x.shape[0]
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(i+1, num_nodes)], dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)
    data_list.append(data)


batch_size = 1
dataloader = DataLoader(data_list, batch_size=batch_size,shuffle=False)

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(1, 1)
        self.conv2 = GCNConv(1, 1)
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation
        
        
    def forward(self, data):
        #print("GCN input")
        x, edge_index = data.x, data.edge_index
        #print(x.size(),edge_index.size())
        x=self.conv1(x, edge_index)
        #print(x.size())
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        #print(x.size())
        x = self.linear(x.T)
        #print(x.size())
        #x = self.sigmoid(x)  # Apply sigmoid activation
        return x

input_dim = data_list[0].x.size(0)
hidden_dim = 64
output_dim = 1

model = GCNModel(input_dim, hidden_dim, output_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    for batch_data in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_data)

        loss = criterion(outputs, batch_data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

print("Training finished.")

# 모델 평가
model.eval()  # 모델을 평가 모드로 변경

correct = 0
total = 0

# 평가 데이터셋에 대해 예측 및 정확도 계산
with torch.no_grad():
    for batch_data in dataloader:
        outputs = model(batch_data)
        predicted = (outputs > 0.5).float()  # 예측값을 0.5를 기준으로 이진 분류
        
        total += batch_data.y.size(0)
        correct += (predicted == batch_data.y.view(-1, 1)).sum().item()

accuracy = correct / total




print(f"Accuracy on the training dataset: {accuracy:.2%}")



# # Visualize some of the original data points
# num_samples_to_visualize = 3  # Change this value as needed
# for i in range(num_samples_to_visualize):
#     sample_data = data_samples_from_images[i]
    
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(131)
#     plt.imshow(sample_data['image'], cmap='gray')
#     plt.title("Image")
    
#     plt.subplot(132)
#     plt.text(0.5, 0.5, sample_data['text'], fontsize=12, ha='center', va='center')
#     plt.title("Text (Sample)")
#     plt.axis('off')
    
#     plt.subplot(133)
#     plt.bar(range(len(sample_data['numeric'])), sample_data['numeric'])
#     plt.title("Numeric Features")
    
#     plt.tight_layout()
#     plt.show()
# # Visualize the graph structure


# Visualize the graph structure and original data
num_samples_to_visualize = 3  # Change this value as needed

for i in range(num_samples_to_visualize):
    sample_data = data_samples_from_images[i]
    graph_data = data_list[i]  # Corresponding graph data
    
    edge_index = graph_data.edge_index.numpy()
    num_nodes = graph_data.x.size(0)
    
    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.T)
    
    # Define node types for original data
    num_image_nodes = image_size * image_size
    num_text_nodes = len(sample_data['text'])
    node_types = ['image'] * (image_size * image_size) + ['text'] * desired_text_feature_length + ['numeric'] * 12
    
    # Define colors for different node types
    node_colors = {
        'image': 'lightblue',
        'text': 'lightgreen',
        'numeric': 'lightsalmon'
    }
    
    plt.figure(figsize=(15, 8))
    
    # Plot graph structure
    plt.subplot(121)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_size=200, node_color=[node_colors[node_types[idx]] for idx in range(num_nodes)])

    # Create a legend for node types
    plt.scatter([], [], c=node_colors['image'], label='Image')
    plt.scatter([], [], c=node_colors['text'], label='Text')
    plt.scatter([], [], c=node_colors['numeric'], label='Numeric')
    plt.legend()
    
    plt.title("Graph Structure")
    
    # Plot original data
    plt.subplot(122)
    plt.imshow(sample_data['image'], cmap='gray')
    plt.title("Image")
    
    plt.tight_layout()
    plt.show()
