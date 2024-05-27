import numpy as np
import torch
from torch.utils.data import DataLoader
from algorithms.utils.DataLoader_MS import MSTimeSeriesData
from algorithms.models.IRMSTransformer import IRMSTransformer
from algorithms.models.loss.FocalLoss import FocalLoss
from algorithms.utils.EarlyStop import EarlyStopping
from algorithms.utils.utils import calculate_model_accuracy

def main(train_data_dir, val_data_dir, test_data_dir, model_save_path, output_widget=None):
    # 设置设备
    # output_widget.setText("训练中...")
    # print(output_widget)
    # print(output_widget is not None)
    # return None
    if output_widget is not None:
        output_widget.setText("加载数据...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载训练数据
    data_loader = DataLoader(dataset=MSTimeSeriesData(S1_data_path=train_data_dir, S1_input_bands=["VV", "VH"],
                                                          S2_data_path=train_data_dir, S2_input_bands= ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', "DOY"]),
                             batch_size=128,
                             shuffle=True)
    # 加载验证数据
    val_data_loader = DataLoader(dataset=MSTimeSeriesData(S1_data_path=val_data_dir, S1_input_bands=["VV", "VH"],
                                                          S2_data_path=val_data_dir, S2_input_bands= ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', "DOY"]),
                             batch_size=128,
                             shuffle=True)
    # 加载测试数据
    test_data_loader = DataLoader(dataset=MSTimeSeriesData(S1_data_path=test_data_dir, S1_input_bands=["VV", "VH"],
                                                          S2_data_path=test_data_dir, S2_input_bands= ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', "DOY"]),
                             batch_size=128,
                             shuffle=True)

    # 创建模型
    # transformer
    print("Creating model...")
    if output_widget is not None:
        output_widget.setText(output_widget.text()+"\n创建模型...")
    model = IRMSTransformer(
        S1_encoder_config={
            "in_channels": 2,
            "hidden_dim": 64,
            "seq_length": 12,
            "d_model": 64,
            "n_head": 4,
            "d_feedforward": 256,
            "n_layers": 2,
            "dropout": 0.4
        },
        S2_encoder_config={
            "in_channels": 10,
            "seq_length": 100,
            "hidden_dim": 64,
            "time_channels": 1,
            "d_model": 64,
            "n_head": 4,
            "d_feedforward": 256,
            "n_layers": 2,
            "dropout": 0.4
        },
        fusion_block_config={
            "d_model": 128,  # 由于两个特征进行了拼接，所以这里的d_model是两个特征的d_model之和
            "n_head": 4,
            "d_feedforward": 256,
            "n_layers": 2,
            "dropout": 0.4,
            "seq_length": 100
        },
        num_classes=3
        ).to(device)
    # 1DCNN
    # model = CNN1D(in_channel=model_config.in_channels, out_channel=model_config.out_channels, dropout=model_config.droupout).to(device)
    # 加载模型参数
    # model.load_state_dict(torch.load(config.model_path))


    # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.4366, 1.4273, 113.7572]).to(device))
    criterion = FocalLoss(gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    earlystop = EarlyStopping(5, verbose=True, path=model_save_path)

    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    # 训练模型
    print("Training model...")
    if output_widget is not None:
        output_widget.setText(output_widget.text()+"\n训练模型...")
    epoches_num = 50
    for epoch in range(epoches_num):
        # 模型训练
        model.train()
        # 记录正确分类的样本数和总样本数
        total_correct_num = 0
        total_num = 0

        for batch_idx, (data, targets) in enumerate(data_loader):
            # 获取数据
            S1, S2 = data[0].to(device), data[1].to(device)
            targets = targets.to(device)
            # 前向传播
            scores = model(S1, S2)
            loss = criterion(scores, targets)

            # 使用混合精度反向传播更新参数
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 计算分类精度
            _, predictions = scores.max(1)
            total_correct_num += (predictions == targets).sum()
            total_num += predictions.size(0)

            # 输出分类精度和平均损失值
            if (batch_idx + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{epoches_num}] Train Loss: {loss.item():.4f}')
                if output_widget is not None:
                    output_widget.setText(output_widget.text() + f'\nEpoch [{epoch + 1}/{epoches_num}] Train Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch + 1}/{epoches_num}] train_acc: {100 * total_correct_num / total_num:.4f}%')
        if output_widget is not None:
            output_widget.setText(output_widget.text() + f'\nEpoch [{epoch + 1}/{epoches_num}] train_acc: {100 * total_correct_num / total_num:.4f}%')

        # 模型验证
        model.eval()
        class_correct_num = [0, 0, 0]
        class_num = [0, 0, 0]
        total_correct_num = 0
        total_num = 0
        with torch.no_grad():
            # 计算验证集的平均损失
            avg_valid_loss = 0
            ground_truth = None
            classification_results = None
            for batch_idx, (data, targets) in enumerate(val_data_loader):
                # 获取数据
                S1, S2 = data[0].to(device), data[1].to(device)
                targets = targets.to(device)
                # 前向传播
                scores = model(S1, S2)
                loss = criterion(scores, targets)
                avg_valid_loss += loss.item()

                # 计算分类精度
                _, predictions = scores.max(1)

                # 保存预测结果和真实标签
                if ground_truth is None:
                    ground_truth = targets.cpu().numpy()
                    classification_results = predictions.cpu().numpy()
                else:
                    ground_truth = np.concatenate((ground_truth, targets.cpu().numpy()))
                    classification_results = np.concatenate((classification_results, predictions.cpu().numpy()))

            # 计算混淆矩阵等评价指标
            metrics = calculate_model_accuracy(ground_truth, classification_results)
            # 打印总体指标
            print("OA:", f"{metrics['accuracy']:.4f}", end='  |  ')
            print("Kappa:", f"{metrics['kappa']:.4f}")
            print(metrics['table_str'])
            if output_widget is not None:
                output_widget.setText(output_widget.text() + f"\nOA: {metrics['accuracy']:.4f}  |  Kappa: {metrics['kappa']:.4f}")
                output_widget.setText(output_widget.text() + f"\n{metrics['table_str']}")

            # 计算平均损失，用于早停判断
            avg_valid_loss /= len(val_data_loader)

        # 早停策略
        earlystop(avg_valid_loss, model)
        if earlystop.early_stop:
            print("此时早停！")
            if output_widget is not None:
                output_widget.setText(output_widget.text() + "\n此时早停！")
            break

if __name__ == "__main__":
    # 训练模型
    datasets = ["20-percent", "40-percent", "60-percent", "80-percent"]
    for dataset in datasets:
        main(train_data_dir="./data/supervised/original_bands/partional_dataset/{dataset}/train",
             val_data_dir="./data/supervised/original_bands/test/",
             test_data_dir="./data/supervised/original_bands/test/",
             model_save_path=f"checkpoints/supervised_IRMSTransformer/irmstransformer_disturbance_type_supervised_{dataset}-dataset.pth")
        print(f"训练集为{dataset}的模型训练完成！")