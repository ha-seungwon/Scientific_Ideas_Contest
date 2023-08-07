import numpy as np
import dgl
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import warnings

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
image_file_paths = ['./data/phone.png', './data/text.png','./data/phone copy.png', './data/text copy.png']

# 데이터 샘플 생성
data_samples_from_images = [generate_sample_from_image(image_path) for image_path in image_file_paths]

num_samples = len(data_samples_from_images)

# 그래프 생성 (간단한 Fully Connected 그래프)
g = dgl.DGLGraph()
g.add_nodes(num_samples)

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
g.add_edges(g.nodes(), g.nodes())

# KMeans 클러스터링
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)  # dtype 설정
cluster_labels = kmeans.fit_predict(data_features.numpy())

# t-SNE 차원 축소
tsne = TSNE(n_components=3, perplexity=3)
visual_features_tsne = tsne.fit_transform(data_features.numpy())

# 클러스터별 3D 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(visual_features_tsne[:, 0], visual_features_tsne[:, 1], visual_features_tsne[:, 2], c=cluster_labels, cmap='rainbow')
legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
ax.add_artist(legend1)
ax.set_title('Clustered Data Points (3D)')
plt.show()

# 새로운 데이터 예측
new_data_features, _ = preprocess_data(data_samples_from_images)
predicted_clusters = kmeans.predict(new_data_features.numpy())

for idx, predicted_cluster in enumerate(predicted_clusters):
    print(f"Predicted Cluster for Image {idx+1}: {predicted_cluster}")
