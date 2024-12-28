import os
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet
import csv

class PatchMaskDataset:
    def __init__(self, root_dir, scanners=None, transform=None):
        """
        root_dir: 数据根目录，按扫描仪组织数据, 如 {scanner}/{image_name}/G4/images。
        scanners: 要加载的扫描仪列表，如 ["Akoya", "KFBio"]。
        transform: 数据变换操作。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        scanners = scanners if scanners else os.listdir(root_dir)  # 如果未指定，默认加载所有扫描仪
        for scanner in scanners:
            scanner_path = os.path.join(root_dir, scanner)
            if os.path.isdir(scanner_path):
                for img_group in os.listdir(scanner_path):
                    img_group_path = os.path.join(scanner_path, img_group)
                    if os.path.isdir(img_group_path):
                        for region in ["G4", "G4_neg"]:
                            images_dir = os.path.join(img_group_path, region, "images")
                            if os.path.exists(images_dir):
                                for patch_file in os.listdir(images_dir):
                                    if patch_file.endswith((".tif", ".tiff")):
                                        img_path = os.path.join(images_dir, patch_file)
                                        self.samples.append((img_path, scanner, img_group, region))

        print(f"Loaded {len(self.samples)} patches from {root_dir} for scanners: {scanners}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, scanner, img_group, region = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, scanner, img_group, region, img_path


def preprocess_features(features):
    """
    对训练特征进行标准化、PCA降维，并计算均值和协方差矩阵。
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_samples, n_features = features_scaled.shape
    n_components = min(n_samples, n_features) - 1
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features_scaled)

    cov_matrix = np.cov(features_reduced, rowvar=False)
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-4  # 稳定协方差矩阵
    cov_inv = np.linalg.pinv(cov_matrix)
    mean = features_reduced.mean(axis=0)

    return scaler, pca, mean, cov_inv


def compute_mahalanobis_for_patches(model, data_loader, scaler_pca_dict, device):
    """
    针对每个补丁计算马氏距离，支持四种特征。
    """
    model.eval()
    patch_distances_dict = {key: [] for key in scaler_pca_dict.keys()}

    with torch.no_grad():
        for images, scanners, img_groups, regions, img_paths in tqdm(data_loader, desc="Processing Patches"):
            images = images.to(device)

            # 提取模型输出，包括中间特征
            logits, first_up, second_up, second_last_down, last_combined = model(images)
            feature_dict = {
                "first_up": first_up.cpu(),
                "second_up": second_up.cpu(),
                "second_last_down": second_last_down.cpu(),
                "last_combined": last_combined.cpu(),
            }

            for i, path in enumerate(img_paths):
                if regions[i] == "G4":  # 只处理 G4 区域
                    for key, features in feature_dict.items():
                        feature = features[i].view(-1).numpy()
                        scaler, pca, train_mean, train_cov_inv = scaler_pca_dict[key]

                        feature_scaled = scaler.transform([feature])
                        feature_reduced = pca.transform(feature_scaled)
                        distance = mahalanobis(feature_reduced[0], train_mean, train_cov_inv)

                        patch_distances_dict[key].append({
                            "scanner": scanners[i],
                            "image_group": img_groups[i],
                            "region": regions[i],
                            "path": path,
                            "distance": distance
                        })

    return patch_distances_dict


if __name__ == "__main__":
    # 数据路径
    train_dir = "/gris/gris-f/homelv/xzhuang/aggc/train_patches3"
    test_dir = "/gris/gris-f/homelv/xzhuang/aggc/test_patches3"
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = UNet(n_channels=3, n_classes=1).to(device)
    checkpoint_path = "/gris/gris-f/homelv/xzhuang/pvc/unet_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model states loaded successfully.")
    else:
        raise FileNotFoundError("Model checkpoint not found.")

    # 提取训练集特征
    train_dataset = PatchMaskDataset(train_dir, scanners=["Akoya"])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    scaler_pca_dict = {}
    model.eval()
    with torch.no_grad():
        for feature_name in ["first_up", "second_up", "second_last_down", "last_combined"]:
            print(f"Processing training features for: {feature_name}")
            train_all_features = []
            for images, scanners, img_groups, regions, img_paths in tqdm(train_loader, desc=f"Extracting {feature_name}"):
                images = images.to(device)
                logits, first_up, second_up, second_last_down, last_combined = model(images)

                feature_dict = {
                    "first_up": first_up.cpu(),
                    "second_up": second_up.cpu(),
                    "second_last_down": second_last_down.cpu(),
                    "last_combined": last_combined.cpu(),
                }

                features = feature_dict[feature_name]
                for i, region in enumerate(regions):
                    if region == "G4":  # 只处理G4区域
                        feature = features[i].view(-1).numpy()
                        train_all_features.append(feature)

            if not train_all_features:
                raise ValueError(f"No G4 features extracted for {feature_name}. Please check the dataset and model.")

            train_all_features = np.vstack(train_all_features)
            scaler, pca, train_mean, train_cov_inv = preprocess_features(train_all_features)
            scaler_pca_dict[feature_name] = (scaler, pca, train_mean, train_cov_inv)
            print(f"Training features processed and PCA fitted for: {feature_name}")

    # 测试集马氏距离计算
    test_dataset = PatchMaskDataset(test_dir, scanners=["Akoya", "KFBio", "Zeiss", "Leica", "Philips", "Olympus"])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    patch_distances_dict = compute_mahalanobis_for_patches(model, test_loader, scaler_pca_dict, device)

    # 保存结果
    for feature_name, patch_distances in patch_distances_dict.items():
        output_path = f"/gris/gris-f/homelv/xzhuang/pvc/patch_mahalanobis_distances_{feature_name}.json"
        with open(output_path, "w") as f:
            json.dump(patch_distances, f, indent=4)
        print(f"Mahalanobis distances for {feature_name} saved to {output_path}")