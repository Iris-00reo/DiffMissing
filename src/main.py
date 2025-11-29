import argparse
import time
import sys
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import yaml
from tqdm import tqdm
import json
sys.path.append("../dataloader")
sys.path.append("../models")
sys.path.append("../utils")
from utils import *
from model import DiffMiss
from mask_metric import masked_mae, masked_rmse, masked_mape
from utils import seed_torch

torch.autograd.set_detect_anomaly(True)

def Inverse_normalization(x, max, min):
    return x * (max - min) + min


def train(model, config, args, train_loader, valid_loader, mask_id, valid_epoch_interval=5, folder_name="", current_time=None):
    optimizer = Adam(params=model.parameters(), lr=config["lr"], weight_decay=1e-6)
    output_path = folder_name + "/model_{}.pth".format(current_time)

    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    best_valid_loss = 1e10
    alpha = config["alpha"]
    beta = config["beta"]
    in_size = config["in_size"]
    device = args.device
    for epoch_idx in range(config["epochs"]):
        avg_loss, avg_pred_loss, avg_noise_loss = 0.0, 0.0, 0.0
        model.train()
        # with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
        #   for batch_idx, train_batch in enumerate(it, start=1):
        for batch_idx, train_batch in enumerate(train_loader):
            optimizer.zero_grad()
            train_feature = train_batch[:, :, :, 0:in_size].to(device)
            train_target = train_batch[:, :, :, -1].to(device)
            noise_loss, pred_target = model(train_feature, mask_id=mask_id, is_train=1)
            pred_target = model.forecast(train_feature, pred_target.contiguous(), mask_id=mask_id, is_train=1)


            pred_loss = masked_mae(pred_target, train_target, 0.0)

            loss = alpha * pred_loss + beta * noise_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            avg_loss += loss.item()
            avg_noise_loss += noise_loss.item()
            avg_pred_loss += pred_loss.item()

        lr_scheduler.step()
        train_loss = avg_loss / len(train_loader)
        train_noise = avg_noise_loss / len(train_loader)
        train_pred_loss = avg_pred_loss / len(train_loader)

        if valid_loader is not None and (epoch_idx + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0.0
            with torch.no_grad():
                # with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                #     for batch_idx, valid_batch in enumerate(it):
                for batch_idx, valid_batch in enumerate(valid_loader):
                    valid_x = valid_batch[:, :, :, 0:in_size].to(device)
                    valid_y = valid_batch[:, :, :, -1].to(device)
                    valid_noise, pred_y = model(valid_x, mask_id=mask_id, is_train=1)
                    pred_y = model.forecast(valid_x, pred_y.contiguous(), mask_id=mask_id, is_train=1)

                    pred_loss = masked_mae(pred_y, valid_y, 0.0)
                    loss = alpha * pred_loss + beta * valid_noise
                    avg_loss_valid += loss.item()
                    # avg_loss_valid += valid_noise.item()

                valid_loss = avg_loss_valid / len(valid_loader)
                print("Epoch {}: train_loss = {} train_noise = {} train_pred_loss = {} valid_loss = {}".format(
                        epoch_idx, train_loss, train_noise, train_pred_loss, valid_loss))
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print("\n best loss is updated to ", avg_loss_valid / len(valid_loader), "at ", epoch_idx)
                torch.save(model.state_dict(), output_path)
        else:
            print("Epoch {}: train_loss = {} train_noise = {} train_pred_loss = {}".format(
                epoch_idx, train_loss, train_noise, train_pred_loss))


def evaluate(model, config, args, test_loader, max_min, mask_id, current_time=None):
    in_size = config["in_size"]
    device = args.device
    with torch.no_grad():
        model.eval()

        all_target = 0.0
        all_pred = 0.0
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_idx, test_batch in enumerate(it, start=1):
                test_feature = test_batch[:, :, :, 0:in_size].to(device)
                test_target = test_batch[:, :, :, -1].to(device)
                _, test_pred = model(test_feature, mask_id=mask_id, is_train=1)
                test_pred = model.forecast(test_feature, test_pred.contiguous(), mask_id=mask_id, is_train=1)

                # # inference
                # test_pred = model.inference(test_feature, n_samples=args.n_sample, is_train=0, batch_idx=batch_idx)
                # test_pred = test_pred.median(dim=1).values

                all_target = torch.cat([all_target, test_target.cpu()], dim=0) if batch_idx != 1 else test_target.cpu()
                all_pred = torch.cat([all_pred, test_pred.cpu()], dim=0) if batch_idx != 1 else test_pred.cpu()

    # print(max_min[0], max_min[1])
    final_target = Inverse_normalization(all_target, max_min[0], max_min[1])
    final_pred = Inverse_normalization(all_pred, max_min[0], max_min[1])
    mae, mape, rmse = (masked_mae(final_pred, final_target, 0.0),
                       masked_mape(final_pred, final_target, 0.0) * 100,
                       masked_rmse(final_pred, final_target, 0.0))
    print('RMSE: {}, MAPE: {}, MAE: {}'.format(rmse, mape, mae))

def main(args):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(current_time)

    seed_torch(args.seed)

    # graph
    # adj_mx, _ = load_adj(args.dataset_path + "/adj_" + args.dataset + ".pkl", "doubletransition")
    # adj_mx = [torch.tensor(i).float() for i in adj_mx]

    # load args
    dataset = args.dataset
    dataset_path = args.dataset_path
    miss_rate = args.missing_ratio
    path = "../config/{}.yaml".format(dataset)
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    batch_size = config["train"]["batch_size"]
    config["model"]["node_num"] = args.node_num
    config["model"]["missing_ratio"] = args.missing_ratio
    config["model"]["seq_len"] = args.seq_len
    print(json.dumps(config, indent=4))

    saving_path = args.saving_path + "/{}/{}".format(dataset, miss_rate)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    # load data
    raw_data = np.load(dataset_path + "data.npz", allow_pickle=True)
    print(raw_data.files)
    mr_str = str("{:.2f}".format(miss_rate))[2:]
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_" + mr_str]), torch.tensor(raw_data["train_y"])], dim=-1).to(torch.float32)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data = torch.cat([torch.tensor(raw_data["valid_x_mask_" + mr_str]), torch.tensor(raw_data["valid_y"])], dim=-1).to(torch.float32)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_" + mr_str]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print("len train dataloader: ", len(train_loader))
    print("len val dataloader: ", len(valid_loader))
    print("len test dataloader: ", len(test_loader))
    mask_id = torch.tensor(raw_data["mask_id_" + mr_str]).to(args.device)
    max_min = raw_data['max_min']

    model = DiffMiss(config, args.device).to(args.device)

    if args.scratch:
        train(model, config["train"], args, train_loader, valid_loader, mask_id=mask_id, folder_name=saving_path, current_time=current_time)
        print("load model from", saving_path)
        model.load_state_dict(torch.load(saving_path + "/model_{}.pth".format(current_time)))
    else:
        print("load model from", args.checkpoint_path)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location='cuda:0'))

    evaluate(model, config["train"], args, test_loader, max_min, mask_id, current_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffMiss")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda_num", default="0")
    parser.add_argument("--dataset", default="PEMS08", type=str, help="dataset name")
    parser.add_argument("--dataset_path", type=str, default="../datasets/PEMS08/")
    parser.add_argument("--saving_path", type=str, default="../saved_models", help="saving model pth")
    parser.add_argument("--seq_len", type=int, default=12, help="sequence length")
    parser.add_argument("--node_num", help="node number", type=int, default=7)
    parser.add_argument("--missing_ratio", type=float, default=0.25, help="missing ratio")
    parser.add_argument("--scratch", action="store_true", help="test or scratch")
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--checkpoint_path", type=str, default="../saved_models/", help="the checkpoint path")
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_num
    start_time = time.time()
    main(args)
    print("Spend Time: ", time.time() - start_time)