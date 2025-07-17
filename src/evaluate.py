from model import LSTMAutoEncoder
from data import TagDataset, df, mean_df, std_df
from train import args, valid_loader
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

def get_reconstruction_loss(args, model, data_loader):
    test_iterator = tqdm(enumerate(data_loader), total=len(data_loader))
    list_losses = []

    with torch.no_grad():
        for i, batch_data in test_iterator:
            batch_data = batch_data.to(args.device)
            predict_values = model(batch_data)

            loss = F.l1_loss(predict_values[0], predict_values[1], reduce=False)
            list_losses.append(loss.mean(dim=1).cpu().numpy())

    return np.concatenate(list_losses, axis=0)

class AnomalyScoreCalculator:
    def __init__(self, mean, std):
        assert mean.shape[0] == std.shape[0] and mean.shape[0] == std.shape[1]
        self.mean = mean
        self.std = std

    def __call__(self, reconstruction_error):
        x = (reconstruction_error - self.mean)
        return np.matmul(np.matmul(x, self.std), x.T)


if __name__ == "__main__":
    model = LSTMAutoEncoder(input_dim=args.input_size, latent_dim=args.latent_size, window_size=args.window_size, num_layers=args.num_layers)
    model.to(args.device)
    model.load_state_dict(torch.load("../results/model.pth"))

    recontruction_loss_normal = get_reconstruction_loss(args, model, valid_loader)
    mean_recontruction_loss_normal = np.mean(recontruction_loss_normal, axis=0)
    std_recontruction_loss_normal = np.cov(recontruction_loss_normal.T)

    anomaly_score_calculator = AnomalyScoreCalculator(mean_recontruction_loss_normal, std_recontruction_loss_normal)
    threshold = 10 # np.quantile([anomaly_score_calculator(i) for i in recontruction_loss_normal], 0.9) # 10
    # np.mean([anomaly_score_calculator(i) for i in recontruction_loss_normal])
    # np.median([anomaly_score_calculator(i) for i in recontruction_loss_normal])

    total_dataset = TagDataset(df=df, input_size=args.input_size, window_size=args.window_size, mean_df=mean_df, std_df=std_df)
    total_data_loader = torch.utils.data.DataLoader(dataset=total_dataset,batch_size=args.batch_size, shuffle=False)
    total_reconstruction_loss = get_reconstruction_loss(args, model, total_data_loader)
    total_anomaly_scores = [anomaly_score_calculator(i) for i in total_reconstruction_loss]

    ##### VISUALIZATION #####
    visualization_df = total_dataset.df
    visualization_df['score'] = total_anomaly_scores
    visualization_df['detection'] = visualization_df['score'] > threshold # 0: NORMAL, 1: ABNORMAL
    # visualization_df['reconstruction_error'] = total_loss.sum(axis=1)

    y_true = (visualization_df['machine_status'] != 'NORMAL').astype(int)
    y_pred = visualization_df['detection']
    print(classification_report(y_true, y_pred))

    plt.figure(figsize=(14, 5))
    plt.ylim(0, 75)
    plt.plot(visualization_df['score'], label='Anomaly Score')
    plt.axhline(y=threshold, color='orange', linestyle='--', label='Threshold')
    # plt.scatter(predicted_anomalies, test_data[predicted_anomalies], color='red', label="Predicted Anomalies")
    # plt.scatter(true_anomalies, test_data[true_anomalies], color='green', marker='x', label="True Anomalies")

    kanban = 1
    ranges_abnormal = []
    for i, j in zip(visualization_df.index, visualization_df['machine_status']):
        if kanban:
            if j != 'NORMAL':
                kanban = 0
                initial_time_point = i
        else:
            if j == 'NORMAL':
                kanban = 1
                ending_time_point = i - 1
                ranges_abnormal.append([initial_time_point, ending_time_point])

    for i, j in enumerate(ranges_abnormal):
        plt.axvspan(j[0], j[1], color='orange', label='True Anomaly' if i==0 else None)

    # visualization_df.iloc[17152, -3]
    # visualization_df.iloc[17153, -3]
    # visualization_df.iloc[18097, -3]
    # visualization_df.iloc[18098, -3]
    #
    # visualization_df.iloc[24507, -3]
    # visualization_df.iloc[24508, -3]
    # visualization_df.iloc[27618, -3]
    # visualization_df.iloc[27619, -3]

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Anomaly Score')
    plt.title("Anomaly Detection Results")
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig('../results/profile_anomaly_scores.png')
    plt.show()