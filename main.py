import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 임의의 그래프를 생성하는 함수 (노드에 속성 추가)
def create_random_graph():
    num_nodes = np.random.randint(5, 15)
    G = nx.erdos_renyi_graph(num_nodes, 0.2)  # 무작위 그래프 생성

    # 노드에 속성 추가
    for node in G.nodes:
        G.nodes[node]['attr'] = {
            'image': create_random_image_data(1),
            'text': create_random_text_data(1),
            'numerical': create_random_numerical_data(1)
        }

    return G

# 임의의 이미지 데이터를 생성하는 함수
def create_random_image_data(num_samples):
    image_data = torch.randn(num_samples, 3, 32, 32)  # 32x32 크기의 RGB 이미지 데이터 생성
    return image_data

# 임의의 텍스트 데이터를 생성하는 함수
def create_random_text_data(num_samples):
    text_data = [f"Text data {i}" for i in range(num_samples)]  # 간단한 텍스트 데이터 생성
    return text_data

# 임의의 수치형 데이터를 생성하는 함수
def create_random_numerical_data(num_samples):
    numerical_data = torch.randn(num_samples, 5)  # 5개의 수치형 특징을 가진 데이터 생성
    return numerical_data

# 각 노드의 속성을 저장하는 클래스
class NodeAttributes(torch.utils.data.Dataset):
    def __init__(self, graph):
        self.graph = graph

    def __len__(self):
        return len(self.graph.nodes)

    def __getitem__(self, idx):
        node = list(self.graph.nodes)[idx]
        img = self.graph.nodes[node]['attr']['image']
        text = self.graph.nodes[node]['attr']['text'][0]
        num = self.graph.nodes[node]['attr']['numerical']

        return (node, img, text, num)

# GNN 모델 정의
class GNNModel(nn.Module):
    def __init__(self, in_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 8)
        self.conv3 = GCNConv(8, 4)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

# 임의의 그래프 생성
graph = create_random_graph()
nx.draw(graph, with_labels=True)
plt.show()

# 데이터 생성
num_samples = graph.number_of_nodes()
dataset = NodeAttributes(graph)

# 데이터로더 생성
loader = DataLoader(dataset, batch_size=num_samples, shuffle=False)

# GNN 모델 생성
image_data = create_random_image_data(num_samples)
text_data = create_random_text_data(num_samples)
numerical_data = create_random_numerical_data(num_samples)
gnn_model = GNNModel(in_channels=image_data.size(1) + 1 + numerical_data.size(1))  # 이미지, 텍스트, 수치형 데이터 크기를 입력 채널로 사용

# 클러스터링을 위한 노드 임베딩 추출
def get_node_embeddings():
    gnn_model.eval()
    with torch.no_grad():
        data = next(iter(loader))
        nodes, images, texts, nums = data

        # 각 노드의 텍스트 데이터를 tensor로 변환하여 unsqueeze 적용
        texts = torch.tensor(texts, dtype=torch.float32)
        texts = texts.unsqueeze(1)

        concatenated_data = torch.cat([images, texts, nums], dim=1)
        edge_index = nx.to_pandas_edgelist(graph).values.T
        embeddings = gnn_model(concatenated_data, torch.tensor(edge_index, dtype=torch.long))
    return embeddings


# 노드 임베딩 추출
embeddings = get_node_embeddings()

# 임베딩을 이용한 클러스터링
from sklearn.cluster import KMeans
print("start kmeans")
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(embeddings.numpy())

# 결과 출력
print("Cluster Labels:")
for node, label in zip(graph.nodes, labels):
    print(f"Node {node}: Cluster {label}")
