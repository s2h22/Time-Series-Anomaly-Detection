import easydict
import torch
from model import LSTMAutoEncoder
from data import TagDataset, normal_df1, normal_df2, mean_df, std_df
from tqdm import tqdm
from torch.nn import functional as F
import numpy as np


def run(args, model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epochs = tqdm(range(args.max_iter // len(train_loader) + 1))

    count = 0
    best_loss = 100000000
    for epoch in epochs:
        model.train()
        optimizer.zero_grad()
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="training")

        for i, batch_data in train_iterator:

            if count > args.max_iter:
                return model
            count += 1

            batch_data = batch_data.to(args.device)
            predict_values = model(batch_data)
            loss = model.loss_function(*predict_values)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_iterator.set_postfix({
                "train_loss": float(loss),
            })

        model.eval()
        eval_loss = 0
        test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="testing")
        with torch.no_grad():
            for i, batch_data in test_iterator:
                batch_data = batch_data.to(args.device)
                predict_values = model(batch_data)
                loss = model.loss_function(*predict_values)

                eval_loss += loss.mean().item()

                test_iterator.set_postfix({
                    "eval_loss": float(loss),
                })
        eval_loss = eval_loss / len(test_loader)
        epochs.set_postfix({
            "Evaluation Score": float(eval_loss),
        })
        if eval_loss < best_loss:
            best_loss = eval_loss
        else:
            if args.early_stop:
                print('early stop condition   best_loss[{}]  eval_loss[{}]'.format(best_loss, eval_loss))
                return model

    return model


args = easydict.EasyDict({
    "batch_size": 128,
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    "input_size": 40,
    "latent_size": 4, # ablation study
    "output_size": 40,
    "window_size" : 10, # ablation study
    "num_layers": 4, # ablation study
    "learning_rate" : 0.001,
    "max_iter" : 100000,
    'early_stop' : True,
})

normal_train = TagDataset(df=normal_df1, input_size=args.input_size, window_size=args.window_size, mean_df=mean_df, std_df=std_df)
normal_threshold = TagDataset(df=normal_df2, input_size=args.input_size, window_size=args.window_size, mean_df=mean_df, std_df=std_df)

# normal_validation = TagDataset(df=normal_df3, input_size=args.input_size, window_size=args.window_size, mean_df=mean_df, std_df=std_df)
# normal_test = TagDataset(df=normal_df4, input_size=args.input_size, window_size=args.window_size, mean_df=mean_df, std_df=std_df)
# abnormal_validation = TagDataset(df=abnormal_df1, input_size=args.input_size, window_size=args.window_size, mean_df=mean_df, std_df=std_df)
# abnormal_test = TagDataset(df=abnormal_df2, input_size=args.input_size, window_size=args.window_size, mean_df=mean_df, std_df=std_df)

train_loader = torch.utils.data.DataLoader(
                 dataset=normal_train,
                 batch_size=args.batch_size,
                 shuffle=True)

valid_loader = torch.utils.data.DataLoader(
                dataset=normal_threshold,
                batch_size=args.batch_size,
                shuffle=False)

model = LSTMAutoEncoder(input_dim=args.input_size, latent_dim=args.latent_size, window_size=args.window_size, num_layers=args.num_layers)
model.to(args.device)
if __name__ == "__main__":
    model = run(args, model, train_loader, valid_loader)
    torch.save(model.state_dict(), "../results/model.pth")