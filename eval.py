import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for anchor, positive, label in data_loader:
            anchor_output = model(anchor)
            positive_output = model(positive)

            # 计算欧式距离
            euclidean_distance = F.pairwise_distance(anchor_output, positive_output)

            # 预测标签（如果距离小于0.5，则认为是正样本对）
            predictions = (euclidean_distance < 0.5).long()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            correct += (predictions == label).sum().item()
            total += label.size(0)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")