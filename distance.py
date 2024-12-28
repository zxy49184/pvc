import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from unet3 import UNet
from torch.utils.data import DataLoader

# Dataset class
class PatchMaskDataset:
    def __init__(self, root_dir, scanners=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        scanners = scanners if scanners else os.listdir(root_dir)
        for scanner in scanners:
            scanner_path = os.path.join(root_dir, scanner)
            if os.path.isdir(scanner_path):
                for img_group in os.listdir(scanner_path):
                    img_group_path = os.path.join(scanner_path, img_group)
                    if os.path.isdir(img_group_path):
                        region = "G4"  # 只处理 G4 区域
                        images_dir = os.path.join(img_group_path, region, "images")
                        if os.path.exists(images_dir):
                            for patch_file in os.listdir(images_dir):
                                if patch_file.endswith((".tif", ".tiff")):
                                    img_path = os.path.join(images_dir, patch_file)
                                    self.samples.append((img_path, scanner, img_group, region))

        print(f"Loaded {len(self.samples)} G4 patches from {root_dir} for scanners: {scanners}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, scanner, img_group, region = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, scanner, img_group, region, img_path


# 提取特征
def extract_all_features(model, data_loader, device):
    model.eval()
    features_dict = {
        "first_up": [],
        "second_up": [],
        "second_last_down": [],
        "last_combined": [],
        "bottleneck": []
    }
    labels = []  # 用于记录扫描仪 ID

    with torch.no_grad():
        for images, scanners, img_groups, regions, img_paths in tqdm(data_loader, desc="Extracting Features"):
            images = images.to(device)
            logits, first_up, second_up, second_last_down, last_combined, bottleneck = model(images)

            for i, scanner in enumerate(scanners):
                features_dict["first_up"].append(first_up[i].detach().cpu().view(-1).numpy())
                features_dict["second_up"].append(second_up[i].detach().cpu().view(-1).numpy())
                features_dict["second_last_down"].append(second_last_down[i].detach().cpu().view(-1).numpy())
                features_dict["last_combined"].append(last_combined[i].detach().cpu().view(-1).numpy())
                features_dict["bottleneck"].append(bottleneck[i].detach().cpu().view(-1).numpy())
                labels.append(scanner)  # 记录扫描仪来源（如 "Akoya", "KFBio"）

    for key in features_dict.keys():
        features_dict[key] = np.vstack(features_dict[key])

    return features_dict, labels


def visualize_tsne_for_scanners_row_layout(train_features_dict, test_features_dict, train_labels, test_labels, save_dir):
    """
    将测试扫描仪与训练集的特征对比，按行排列，每行显示五个图。
    """
    feature_keys = ["first_up", "second_up", "second_last_down", "last_combined", "bottleneck"]
    scanners_to_compare = ["Akoya", "KFBio", "Zeiss", "Leica", "Philips", "Olympus"]

    for feature_key in feature_keys:
        print(f"Performing t-SNE visualization for feature: {feature_key}")

        # 提取特征
        train_features = train_features_dict[feature_key]
        test_features = test_features_dict[feature_key]
        combined_features = np.vstack([train_features, test_features])

        # PCA 降维到 50 维
        pca = PCA(n_components=50, random_state=42)
        combined_features_reduced = pca.fit_transform(combined_features)

        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=15)
        reduced_features = tsne.fit_transform(combined_features_reduced)

        # 设置图像布局
        fig, axes = plt.subplots(1, len(scanners_to_compare), figsize=(25, 5))
        fig.suptitle(f"t-SNE Visualization for {feature_key}", fontsize=16)

        for idx, scanner_name in enumerate(scanners_to_compare):
            ax = axes[idx]

            # 绘制训练集 Akoya 数据 (红色)
            train_akoya_indices = [i for i, label in enumerate(train_labels) if label == "Akoya"]
            ax.scatter(
                reduced_features[train_akoya_indices, 0],
                reduced_features[train_akoya_indices, 1],
                label="Train Akoya",
                color="red",
                alpha=0.6
            )

            # 绘制测试集中该扫描仪的数据 (绿色)
            test_scanner_indices = [
                i + len(train_labels) for i, label in enumerate(test_labels) if label == scanner_name
            ]
            ax.scatter(
                reduced_features[test_scanner_indices, 0],
                reduced_features[test_scanner_indices, 1],
                label=f"Test {scanner_name}",
                color="green",
                alpha=0.6
            )

            # 设置子图标题和图例
            ax.set_title(scanner_name)
            ax.set_xlabel("t-SNE Component 1")
            ax.set_ylabel("t-SNE Component 2")
            ax.legend(loc="best")
            ax.grid()

        # 保存图像
        feature_save_dir = os.path.join(save_dir, feature_key)
        os.makedirs(feature_save_dir, exist_ok=True)
        save_path = os.path.join(feature_save_dir, f"tsne_row_layout_{feature_key}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 保证标题不被覆盖
        plt.savefig(save_path)
        plt.close()
        print(f"t-SNE visualization saved for feature {feature_key} at {save_path}")



def calculate_mahalanobis_distances(train_features, test_features):
    """
    计算马氏距离，用于判断 OOD 数据。
    train_features: 训练集的特征 (N_train, D)
    test_features: 测试集的特征 (N_test, D)
    """
    # 计算训练集的均值和协方差
    train_mean = np.mean(train_features, axis=0)
    train_cov = np.cov(train_features, rowvar=False)
    train_cov_inv = np.linalg.inv(train_cov + np.eye(train_cov.shape[0]) * 1e-5)  # 防止奇异矩阵

    # 计算测试集中每个样本的马氏距离
    distances = []
    for feature in test_features:
        diff = feature - train_mean
        distance = np.sqrt(np.dot(np.dot(diff.T, train_cov_inv), diff))
        distances.append(distance)

    return np.array(distances)


def visualize_mahalanobis_distances(train_features_dict, test_features_dict, train_labels, test_labels, save_dir):

    feature_keys = ["first_up", "second_up", "second_last_down", "last_combined", "bottleneck"]

    for feature_key in feature_keys:
        print(f"Calculating Mahalanobis distances for feature: {feature_key}")

        # 提取训练集和测试集的特征
        train_features = train_features_dict[feature_key]
        test_features = test_features_dict[feature_key]

        # 计算马氏距离
        distances = calculate_mahalanobis_distances(train_features, test_features)

        # 分别绘制训练集和测试集的马氏距离分布
        plt.figure(figsize=(10, 6))
        plt.hist(distances[:len(train_labels)], bins=50, alpha=0.6, label="Train", color="blue")
        plt.hist(distances[len(train_labels):], bins=50, alpha=0.6, label="Test", color="red")
        plt.xlabel("Mahalanobis Distance")
        plt.ylabel("Frequency")
        plt.title(f"Mahalanobis Distance Distribution for {feature_key}")
        plt.legend(loc="upper right")
        plt.grid()

        # 保存图像
        feature_save_dir = os.path.join(save_dir, feature_key)
        os.makedirs(feature_save_dir, exist_ok=True)
        save_path = os.path.join(feature_save_dir, f"mahalanobis_{feature_key}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Mahalanobis distance visualization saved for {feature_key} at {save_path}")


if __name__ == "__main__":
    # 数据路径
    train_dir = "/gris/gris-f/homelv/xzhuang/aggc/train_patches2"
    test_dir = "/gris/gris-f/homelv/xzhuang/aggc/test_patches2"
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    save_dir = "/gris/gris-f/homelv/xzhuang/pvc/visualizations"
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    model = UNet(n_channels=3, n_classes=1).to(device)
    checkpoint_path = "/gris/gris-f/homelv/xzhuang/pvc/unet_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model states loaded successfully.")
    else:
        raise FileNotFoundError("Model checkpoint not found.")

    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # 加载训练集和测试集
    train_dataset = PatchMaskDataset(train_dir, scanners=["Akoya"], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    test_dataset = PatchMaskDataset(test_dir, scanners=["Akoya", "KFBio", "Zeiss", "Leica", "Philips", "Olympus"], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 提取特征
    train_features_dict, train_labels = extract_all_features(model, train_loader, device)
    test_features_dict, test_labels = extract_all_features(model, test_loader, device)

    # 只调用一次可视化函数
    print("Visualizing t-SNE for all features...")
    visualize_tsne_for_scanners_row_layout(
        train_features_dict, test_features_dict, train_labels, test_labels, save_dir
    )
    # 可视化马氏距离分布
    visualize_mahalanobis_distances(train_features_dict, test_features_dict, train_labels, test_labels, save_dir)
