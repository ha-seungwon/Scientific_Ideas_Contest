import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import warnings
import dgl.function as fn
import dgl.nn as dglnn


warnings.filterwarnings('ignore')
# 이미지 크기와 샘플 개수 설정

image_size = 32
# 데이터 샘플 생성 함수
def generate_sample_from_image(image_path):
    image = Image.open(image_path)
    image = image.resize((image_size, image_size))
    image = np.array(image)
    text = np.random.choice(['spam', 'ham', 'egg'], size=1)[0]
    numeric = np.random.rand(4)
    
    return {'image': image, 'text': text, 'numeric': numeric}

# 이미지 파일 경로
image_file_paths = ['./data/phone.png', './data/text.png','./data/phone copy.png', './data/text copy.png',
                    './data/phone.png', './data/text.png','./data/phone copy.png', './data/text copy.png',
                    './data/phone.png', './data/text.png','./data/phone copy.png', './data/text copy.png',
                    './data/phone.png', './data/text.png','./data/phone copy.png', './data/text copy.png',
                    './data/phone.png', './data/text.png','./data/phone copy.png', './data/text copy.png',
                    './data/phone.png', './data/text.png','./data/phone copy.png', './data/text copy.png']

# 데이터 샘플 생성
data_samples_from_images = [generate_sample_from_image(image_path) for image_path in image_file_paths]

num_samples = len(data_samples_from_images)



# 데이터 전처리 및 그래프 생성
def preprocess_data(data_samples):
    all_features = []
    for sample in data_samples:
        image_feature = sample['image'].reshape(-1)
        text_feature = np.array([1 if sample['text'] == 'spam' else 0,
                                 1 if sample['text'] == 'ham' else 0,
                                 1 if sample['text'] == 'egg' else 0])
        numeric_feature = sample['numeric']
        combined_feature = np.concatenate((image_feature, text_feature, numeric_feature))
        all_features.append(combined_feature)
    data_feature_len = len(combined_feature)  # 계산된 총 특성의 길이
    return torch.FloatTensor(all_features), data_feature_len

data_features, data_feature_len = preprocess_data(data_samples_from_images)


print(data_features.size(), data_feature_len )
# 그래프 생성
g = dgl.DGLGraph()  # 빈 그래프 생성

# 노드 추가
g.add_nodes(num_samples)  # add_nodes 메소드 사용

# 각 데이터 샘플의 인덱스를 연결하여 엣지 추가
for i in range(num_samples):
    for j in range(num_samples):
        if i != j:
            g.add_edges(i, j)
# 레이블 생성
labels = torch.tensor([0, 1, 0, 1] * (num_samples // 4))  # num_samples 만큼 레이블을 반복

import torch.nn.functional as F

class GraphSAGENet(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, aggregator_type='mean'):
        super(GraphSAGENet, self).__init__()
        self.sage = dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type)
        self.fc = nn.Linear(hidden_feats, num_classes)

    def forward(self, g, features):
        h = self.sage(g, features)
        h = F.relu(h)
        h = self.fc(h)
        return h
# 모델 및 손실 함수, 옵티마이저 초기화
model = GraphSAGENet(in_feats=1031, hidden_feats=16, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(g, data_features)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

print("Training finished!")

# 임베딩 추출 (t-SNE 차원 축소)
with torch.no_grad():
    model.eval()
    embeddings = model(g, data_features).detach().numpy()







validation_data_samples = [generate_sample_from_image(image_file_paths[0])]  # 검증 데이터 생성
validation_data_features, _ = preprocess_data(validation_data_samples)  # 데이터 전처리

print(validation_data_features.size())  # 검증 데이터의 특성 크기 출력

# 그래프 생성
g_predict = dgl.DGLGraph()  # 빈 그래프 생성

# 노드 추가
g_predict.add_nodes(validation_data_features.size(0))  # 검증 데이터 노드 수와 일치하도록 수정

# 각 데이터 샘플의 인덱스를 연결하여 엣지 추가
for i in range(validation_data_features.size(0)):
    for j in range(validation_data_features.size(0)):
        if i != j:
            g_predict.add_edges(i, j)

# 검증 데이터에 대한 예측
with torch.no_grad():
    validation_outputs = model(g_predict, validation_data_features)

# validation_outputs를 통해 예측 결과를 얻을 수 있습니다.
predicted_labels = torch.argmax(validation_outputs, dim=1)

print("Predicted labels for validation data:", predicted_labels)








# t-SNE 차원 축소
tsne = TSNE(n_components=3, perplexity=3)
visual_features_tsne = tsne.fit_transform(data_features.numpy())

# 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(visual_features_tsne[:, 0], visual_features_tsne[:, 1], visual_features_tsne[:, 2], c=labels, cmap='rainbow')
legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
ax.add_artist(legend1)
ax.set_title('Clustered Data Points (3D)')
plt.show()