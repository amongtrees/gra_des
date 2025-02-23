import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from eval import evaluate
from model import config
from model.train import train
from model.transformer import TransformerModel
from loader import load_csv, ContrastiveDataset

if __name__ == '__main__':
    # 加载数据
    doh_data = load_csv('../CSVs/Total_CSVs/l1-doh.csv')
    nondoh_data = load_csv('../CSVs/Total_CSVs/l1-nondoh.csv')

    # 划分 DoH 和 Non-DoH 数据集
    doh_train, doh_test = train_test_split(doh_data, test_size=0.2, random_state=42)
    nondoh_train, nondoh_test = train_test_split(nondoh_data, test_size=0.2, random_state=42)

    # 创建训练集和测试集的 ContrastiveDataset
    train_dataset = ContrastiveDataset(doh_train, nondoh_train)
    test_dataset = ContrastiveDataset(doh_test, nondoh_test)

    # 创建数据加载器
    train_data_loader = DataLoader(train_dataset, batch_size=config.DATA_LOADER_CONFIG["batch_size"], shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=config.DATA_LOADER_CONFIG["batch_size"], shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(input_size=doh_data.shape[1], hidden_dim=config.MODEL_CONFIG["hidden_dim"],
                         num_layers=config.MODEL_CONFIG["num_layers"],
                         nhead=config.MODEL_CONFIG["nhead"]).to(device)
    train(model,device,train_data_loader,test_data_loader)
    evaluate(model, test_data_loader)
    torch.save(model, "is_non_model.pth")