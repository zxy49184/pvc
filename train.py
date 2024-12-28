import os
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from unet3 import UNet
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Sampler, Dataset
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Subset


class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None, neg_sample_ratio=1.0):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.positive_samples = []
        self.negative_samples = []

        for img_group in os.listdir(root_dir):
            img_group_path = os.path.join(root_dir, img_group)
            if os.path.isdir(img_group_path):
                for region in ["G4", "G4_neg"]:  # G4为正样本，G4_neg为负样本
                    images_dir = os.path.join(img_group_path, region, "images")
                    masks_dir = os.path.join(img_group_path, region, "masks")
                    if os.path.exists(images_dir) and os.path.exists(masks_dir):
                        for patch_file in os.listdir(images_dir):
                            if patch_file.endswith((".tif", ".tiff")):
                                image_path = os.path.join(images_dir, patch_file)
                                mask_path = os.path.join(masks_dir, patch_file)
                                if os.path.exists(mask_path):
                                    label = 1 if region == "G4" else 0
                                    sample = (image_path, mask_path, label)
                                    if label == 1:
                                        self.positive_samples.append(sample)
                                    else:
                                        self.negative_samples.append(sample)

        # 随机采样负样本，确保与正样本比例匹配
        neg_sample_count = int(len(self.positive_samples) * neg_sample_ratio)
        self.negative_samples = random.sample(self.negative_samples, min(neg_sample_count, len(self.negative_samples)))
        self.samples = self.positive_samples + self.negative_samples
        random.shuffle(self.samples)  # 混合正负样本，确保随机性

        print(f"Loaded {len(self.samples)} patches: {len(self.positive_samples)} positives and {len(self.negative_samples)} negatives.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            transformed = self.transform({"image": image, "mask": mask})
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask, label


def split_dataset_balanced(dataset, train_ratio=0.8, seed=42):

    # 设置随机种子，确保结果可复现
    random.seed(seed)

    # 获取正负样本的索引
    positive_indices = [i for i, sample in enumerate(dataset.samples) if sample[2] == 1]
    negative_indices = [i for i, sample in enumerate(dataset.samples) if sample[2] == 0]

    # 打乱索引
    random.shuffle(positive_indices)
    random.shuffle(negative_indices)

    # 按比例划分正负样本
    pos_train_size = int(train_ratio * len(positive_indices))
    neg_train_size = int(train_ratio * len(negative_indices))

    train_indices = positive_indices[:pos_train_size] + negative_indices[:neg_train_size]
    val_indices = positive_indices[pos_train_size:] + negative_indices[neg_train_size:]

    # 打乱训练集和验证集的索引
    random.shuffle(train_indices)
    random.shuffle(val_indices)

    # 构建训练集和验证集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


class BalancedBatchSampler(Sampler):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # Access the underlying dataset if it's a Subset
        if isinstance(dataset, torch.utils.data.Subset):
            indices = dataset.indices  # Indices of the Subset
            full_samples = dataset.dataset.samples  # Access the original dataset's samples
            self.positive_indices = [i for i in indices if full_samples[i][2] == 1]
            self.negative_indices = [i for i in indices if full_samples[i][2] == 0]
        else:
            self.positive_indices = [i for i, sample in enumerate(dataset.samples) if sample[2] == 1]
            self.negative_indices = [i for i, sample in enumerate(dataset.samples) if sample[2] == 0]

    def __iter__(self):
        random.shuffle(self.positive_indices)
        random.shuffle(self.negative_indices)

        positive_batch_size = self.batch_size // 2
        negative_batch_size = self.batch_size - positive_batch_size

        batches = []
        for i in range(0, min(len(self.positive_indices), len(self.negative_indices)), positive_batch_size):
            pos_batch = self.positive_indices[i:i + positive_batch_size]
            neg_batch = self.negative_indices[i:i + negative_batch_size]

            if len(pos_batch) < positive_batch_size:
                pos_batch += random.sample(self.positive_indices, positive_batch_size - len(pos_batch))
            if len(neg_batch) < negative_batch_size:
                neg_batch += random.sample(self.negative_indices, negative_batch_size - len(neg_batch))

            batch = pos_batch + neg_batch
            random.shuffle(batch)
            batches.append(batch)

        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size



def smooth_curve(values, weight=0.85):

    smoothed = []
    last = values[0]  # 初始值
    for val in values:
        smoothed_val = last * weight + (1 - weight) * val  # EMA公式
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# 自定义变换
class JointTransform:
    def __init__(self, image_transform=None, mask_transform=None, augment=False):
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augment = augment

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return {"image": image, "mask": mask}


# Dice 系数计算
def dice_coeff(input, target, epsilon=1e-6):
    input = (input > 0.5).float()
    intersection = (input * target).sum(dim=(1, 2, 3))
    union = input.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

# 定义组合损失函数
def combined_loss(outputs, masks):
    bce_loss = nn.BCEWithLogitsLoss()(outputs, masks)
    dice = DiceLoss()(outputs, masks)  # 使用自定义的 Dice 损失函数
    return bce_loss, dice  # 返回两个单独的损失值

def train(model, train_loader, optimizer, device):
    model.train()
    epoch_bce_loss = 0
    epoch_dice_loss = 0
    epoch_total_loss = 0
    epoch_dice_coeff = 0
    total_batches = len(train_loader)

    for images, masks, _ in tqdm(train_loader, desc="Training"):
        images = images.to(device)  # (B, C, H, W)
        masks = masks.to(device)    # (B, 1, H, W)

        optimizer.zero_grad()
        outputs, _, _, _, _, _ = model(images)  # outputs shape: (B, 1, H, W)

        # 计算 BCE 和 Dice 损失
        bce_loss, dice_loss = combined_loss(outputs, masks)
        total_loss = 0.7 * bce_loss + 0.3 * dice_loss  # 加权总损失

        # 反向传播与优化
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸
        optimizer.step()

        # 累加每个 batch 的损失
        epoch_bce_loss += bce_loss.item()
        epoch_dice_loss += dice_loss.item()
        epoch_total_loss += total_loss.item()

        # 累加 Dice 系数
        outputs = torch.sigmoid(outputs)  # 转换为概率
        epoch_dice_coeff += dice_coeff(outputs, masks)  # 计算 Dice 系数

    # 计算每个 epoch 的平均 BCE Loss、Dice Loss、总损失和 Dice 系数
    avg_bce_loss = epoch_bce_loss / total_batches
    avg_dice_loss = epoch_dice_loss / total_batches
    avg_total_loss = epoch_total_loss / total_batches
    avg_dice_coeff = epoch_dice_coeff / total_batches

    return avg_bce_loss, avg_dice_loss, avg_total_loss, avg_dice_coeff


@torch.inference_mode()
def validate(model, val_loader, device):
    model.eval()
    val_bce_loss = 0
    val_dice_loss = 0
    val_total_loss = 0
    val_dice_coeff = 0
    total_batches = len(val_loader)

    for images, masks, _ in tqdm(val_loader, desc="Validating"):
        images = images.to(device)  # (B, C, H, W)
        masks = masks.to(device)    # (B, 1, H, W)

        outputs, _, _, _, _, _ = model(images)

        # 计算 BCE 和 Dice 损失
        bce_loss, dice_loss = combined_loss(outputs, masks)
        total_loss = 0.7 * bce_loss + 0.3 * dice_loss  # 加权总损失

        # 累加每个 batch 的损失
        val_bce_loss += bce_loss.item()
        val_dice_loss += dice_loss.item()
        val_total_loss += total_loss.item()

        # 累加 Dice 系数
        outputs = torch.sigmoid(outputs)
        val_dice_coeff += dice_coeff(outputs, masks)

    # 计算每个 epoch 的平均 BCE Loss、Dice Loss、总损失和 Dice 系数
    avg_bce_loss = val_bce_loss / total_batches
    avg_dice_loss = val_dice_loss / total_batches
    avg_total_loss = val_total_loss / total_batches
    avg_dice_coeff = val_dice_coeff / total_batches

    return avg_bce_loss, avg_dice_loss, avg_total_loss, avg_dice_coeff



def wrap_dataloader(data_loader):

    for image, mask, label in data_loader:
        # 模拟 scanner 和 img_group 信息
        scanner = ["Simulated_Scanner"] * image.size(0)
        img_group = ["Simulated_Group"] * image.size(0)
        region = ["G4" if lbl == 1 else "G4_neg" for lbl in label]
        img_path = [f"Simulated_Path_{i}" for i in range(image.size(0))]
        yield image, scanner, img_group, region, img_path


# 绘制训练和验证曲线
def plot_metrics(train_losses, val_losses, train_dices, val_dices, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    # 对数据应用平滑
    smooth_train_losses = smooth_curve(train_losses)
    smooth_val_losses = smooth_curve(val_losses)
    smooth_train_dices = smooth_curve(train_dices)
    smooth_val_dices = smooth_curve(val_dices)

    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, smooth_train_losses, 'r-', label="Train Loss (Smoothed)")  # 平滑后训练损失
    plt.plot(epochs, smooth_val_losses, 'g-', label="Val Loss (Smoothed)")  # 平滑后验证损失
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    # Dice 系数曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, smooth_train_dices, 'r-', label="Train Dice (Smoothed)")  # 平滑后训练Dice
    plt.plot(epochs, smooth_val_dices, 'g-', label="Val Dice (Smoothed)")  # 平滑后验证Dice
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.legend()
    plt.title("Dice Curve")

    plt.tight_layout()

    # 保存或展示图像
    if save_path:
        plt.savefig(save_path)
        print(f"Metrics plot saved to {save_path}")
    plt.show()  # 保存图片后，显示图片

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    将标准化的图像数据还原到 [0, 1] 范围
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1)  # (C, 1, 1) 扩展为对应维度
    std = torch.tensor(std).view(1, 3, 1, 1)
    tensor = tensor * std + mean  # 反标准化
    return torch.clamp(tensor, 0, 1)  # 限制到 [0, 1] 范围内

def visualize_predictions(model, loader, device, save_dir, num_images=5, set_name="val"):
    """
    可视化模型的预测结果，并保存到指定目录。
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    count = 0

    with torch.no_grad():
        for images, masks, labels in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs, *_ = model(images)
            preds = torch.sigmoid(outputs) > 0.5

            # 反归一化图像
            images_denorm = denormalize(images.cpu())

            for i in range(images.size(0)):
                if count >= num_images:
                    return
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(images_denorm[i].permute(1, 2, 0))  # 反归一化后的输入图像
                ax[0].set_title(f"{set_name.capitalize()} - Input Image")
                ax[1].imshow(masks[i].cpu().squeeze(), cmap='gray')
                ax[1].set_title(f"{set_name.capitalize()} - Ground Truth")
                ax[2].imshow(preds[i].cpu().squeeze(), cmap='gray')
                ax[2].set_title(f"{set_name.capitalize()} - Prediction")

                save_path = os.path.join(save_dir, f"{set_name}_sample_{count}.png")
                plt.savefig(save_path)
                plt.close()
                count += 1

def extract_features(model, data_loader, device):
    model.eval()
    features_dict = {
        "first_up": [],
        "second_up": [],
        "second_last_down": [],
        "last_combined": [],
        "bottleneck": []  # 新增瓶颈特征
    }
    labels = []

    with torch.no_grad():
        for images, scanners, img_groups, regions, img_paths in tqdm(data_loader, desc="Extracting Features"):
            images = images.to(device)

            # 提取模型输出，包括中间特征
            logits, first_up, second_up, second_last_down, last_combined, bottleneck = model(images)

            for i, region in enumerate(regions):
                # 仅处理 G4 区域
                if region == "G4":
                    # 提取并存储每种特征
                    features_dict["first_up"].append(first_up[i].detach().cpu().view(-1).numpy())
                    features_dict["second_up"].append(second_up[i].detach().cpu().view(-1).numpy())
                    features_dict["second_last_down"].append(second_last_down[i].detach().cpu().view(-1).numpy())
                    features_dict["last_combined"].append(last_combined[i].detach().cpu().view(-1).numpy())
                    features_dict["bottleneck"].append(bottleneck[i].detach().cpu().view(-1).numpy())  # 添加瓶颈特征
                    # 记录对应的标签
                    labels.append(f"{scanners[i]}_{region}")  # 用于区分训练和 OOD 数据

    # 将特征转换为数组
    for key in features_dict.keys():
        features_dict[key] = np.vstack(features_dict[key])

    return features_dict, labels

def visualize_tsne_for_all_features(train_features_dict, train_labels,
                                    val_features_dict, val_labels, epoch, save_dir):
    """
    可视化训练集和验证集在 5个不同特征上的 t-SNE 分布。
    """
    # 特征列表
    feature_keys = ["first_up", "second_up", "second_last_down", "last_combined", "bottleneck"]

    for feature_key in feature_keys:
        print(f"Performing t-SNE visualization for feature: {feature_key} at epoch {epoch}")

        # 合并训练集和验证集的特征和标签
        all_features = np.vstack([train_features_dict[feature_key], val_features_dict[feature_key]])
        all_labels = np.array(train_labels + val_labels)

        # 生成一个数组用于区分训练集和验证集
        data_source = ["Train"] * len(train_labels) + ["Validation"] * len(val_labels)

        # 执行 t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_features = tsne.fit_transform(all_features)

        # 绘制 t-SNE 可视化图
        plt.figure(figsize=(10, 8))
        for source, color in zip(["Train", "Validation"], ["blue", "red"]):
            indices = [i for i, s in enumerate(data_source) if s == source]
            plt.scatter(
                reduced_features[indices, 0],
                reduced_features[indices, 1],
                label=source,
                alpha=0.6,
                c=color
            )

        plt.title(f"t-SNE Visualization for {feature_key} at Epoch {epoch}")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend(loc="best", fontsize="small")
        plt.grid()

        # 保存图像
        feature_save_dir = os.path.join(save_dir, feature_key)
        os.makedirs(feature_save_dir, exist_ok=True)
        save_path = os.path.join(feature_save_dir, f"tsne_epoch_{epoch}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"t-SNE visualization saved for feature {feature_key} at {save_path}")


# 主程序
if __name__ == "__main__":
    train_dir = "/gris/gris-f/homelv/xzhuang/aggc/train_patches/Akoya"
    #val_dir = "/gris/gris-f/homelv/xzhuang/aggc/test_patches2/Akoya"
    epochs = 150
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # 数据增强
    train_transform = JointTransform(
        image_transform=transforms.Compose([
            # 随机水平翻转
            transforms.RandomHorizontalFlip(p=0.5),
            # 随机垂直翻转
            transforms.RandomVerticalFlip(p=0.5),
            # 转换为Tensor
            transforms.ToTensor(),
            # 归一化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        mask_transform=transforms.Compose([
            # 随机水平翻转和垂直翻转需与图像保持一致
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # 随机裁剪
            transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
            # 转换为Tensor
            transforms.ToTensor()
        ]),
        augment=True
    )
    val_transform = JointTransform(
        image_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        mask_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        augment=False
    )

    full_dataset = PatchDataset(train_dir, transform=train_transform)

    # 平衡划分数据集
    train_dataset, val_dataset = split_dataset_balanced(full_dataset, train_ratio=0.8, seed=42)

    # 设置 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    save_dir = "/gris/gris-f/homelv/xzhuang/pvc/visualizations/tsne_features"

    # 初始化模型
    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=7)
    # 损失函数
    criterion = combined_loss
    # 加载已有的模型权重
    checkpoint_path = "/gris/gris-f/homelv/xzhuang/pvc/unet_model.pth"
    # 加载模型、优化器和调度器状态
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model, optimizer, and scheduler states loaded successfully.")
    else:
        print("No existing model found. Starting training from scratch.")


    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_bce_loss, train_dice_loss, train_total_loss, train_dice = train(model, train_loader, optimizer, device)
        val_bce_loss, val_dice_loss, val_total_loss, val_dice = validate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"Train - BCE Loss: {train_bce_loss:.4f}, Dice Loss: {train_dice_loss:.4f}, Total Loss: {train_total_loss:.4f}, Dice Coefficient: {train_dice:.4f}")
        print(
            f"Val - BCE Loss: {val_bce_loss:.4f}, Dice Loss: {val_dice_loss:.4f}, Total Loss: {val_total_loss:.4f}, Dice Coefficient: {val_dice:.4f}")

        train_losses.append(train_total_loss)
        val_losses.append(val_total_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        # 保存最佳模型
        if val_total_loss < best_val_loss :
            best_val_loss = val_total_loss
            # 保存模型与优化器状态
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "/gris/gris-f/homelv/xzhuang/pvc/unet_model.pth")

            print(f"Epoch {epoch + 1}: Improved validation loss. Model saved.")

        scheduler.step(val_total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}: Learning Rate: {current_lr:.6f}")

        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch + 1}: Visualizing predictions...")

            # 可视化验证集
            val_save_dir = f"/gris/gris-f/homelv/xzhuang/pvc/predictions2/val/epoch_{epoch + 1}"
            visualize_predictions(model, val_loader, device, save_dir=val_save_dir, num_images=5, set_name="val")
            print(f"Validation predictions for epoch {epoch + 1} saved to {val_save_dir}.")

            # 可视化训练集
            train_save_dir = f"/gris/gris-f/homelv/xzhuang/pvc/predictions2/train/epoch_{epoch + 1}"
            visualize_predictions(model, train_loader, device, save_dir=train_save_dir, num_images=5, set_name="train")
            print(f"Training predictions for epoch {epoch + 1} saved to {train_save_dir}.")


        # if (epoch + 1) % 50 == 0:
        #     print(f"Performing t-SNE visualization at epoch {epoch + 1}")
        #     train_features_dict, train_labels = extract_features(model, wrap_dataloader(train_loader), device)
        #     val_features_dict, val_labels = extract_features(model, wrap_dataloader(val_loader), device)
        #
        #     # 保存 4 个特征的 t-SNE 可视化图像
        #     visualize_tsne_for_all_features(
        #         train_features_dict, train_labels,
        #         val_features_dict, val_labels,
        #         epoch + 1,
        #         save_dir=os.path.join(save_dir, f"tsne_visualizations")
        #     )

    # 绘制训练和验证曲线
    save_path = "/gris/gris-f/homelv/xzhuang/pvc/training_validation_metrics1.png"
    plot_metrics(train_losses, val_losses, train_dices, val_dices, save_path)
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")



