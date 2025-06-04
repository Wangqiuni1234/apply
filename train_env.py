import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models.mymodel import CombineModel
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpeechEmotionDataset(Dataset):
    """自定义数据集类，用于加载数据"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return torch.tensor(self.features[item], dtype=torch.float32), torch.tensor(self.labels[item], dtype=torch.long)

def load_FL(features_dir):
    """加载特征和对应的标签"""
    features = []
    labels = []
    label_map = {'angry': 0, 'happy': 1, 'neutral': 2, 'sad': 3}  # 情感类别标签映射
    for emotion_folder in os.listdir(features_dir):
        emotion_path = os.path.join(features_dir, emotion_folder)
        if not os.path.isdir(emotion_path):
            continue
        label = label_map.get(emotion_folder)  # 从文件夹名称中获取标签
        for features_file in os.listdir(emotion_path):
            if features_file.endswith('.npy'):
                feature_path = os.path.join(emotion_path, features_file)
                feature = np.load(feature_path)  # 加载特征
                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

def plot_confusion_matrix(cm, save_path, class_names):
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.3f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs, save_dir):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total * 100
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.3f}%")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "my_new_best_model.pth"))

    # Evaluate after training
    evaluate(model, test_loader, criterion, save_dir)

def evaluate(model, test_loader, criterion, save_dir):
    model.eval()
    model.load_state_dict(torch.load(os.path.join(save_dir, "my_new_best_model.pth")))
    predicted_labels = []
    true_labels = []
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total * 100
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}%")

    # Classification report
    report = classification_report(true_labels, predicted_labels, target_names=['angry', 'happy', 'neutral', 'sad'])
    print("Classification Report:\n", report)

    # Save classification report
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}%\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:\n", cm)
    plot_confusion_matrix(cm, os.path.join(save_dir, "confusion_matrix.png"), class_names=['angry', 'happy', 'neutral', 'sad'])

if __name__ == '__main__':
    features, labels = load_FL('savedir/saved_features/')
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    trainset = SpeechEmotionDataset(x_train, y_train)
    testset = SpeechEmotionDataset(x_test, y_test)

    train_loader = DataLoader(trainset, batch_size=50, num_workers=0, shuffle=True)
    test_loader = DataLoader(testset, batch_size=50, num_workers=0, shuffle=False)

    model = CombineModel(349, 4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 1000
    save_dir = 'results/'

    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs, save_dir)
