#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import LoadForecaster, LoadForecasterTransformer
from dataset import CustomLoadDataset

forecast_days = 7


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="~/Data/AI-Hero/", type=str)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--save_dir", default=None, help="saves the model, if path is provided")
    parser.add_argument("--historic_window", type=int, default=7 * 24, help="input time steps in hours")
    parser.add_argument("--forecast_horizon", type=int, default=forecast_days * 24, help="forecast time steps in hours")
    parser.add_argument("--hidden_size", type=int, default=48, help="size of the internal state")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--quicktest", action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')

    # Forecast Parameters
    historic_window = args.historic_window
    forecast_horizon = args.forecast_horizon

    # Loading Data
    data_dir = args.data_dir
    train_set = CustomLoadDataset(
        os.path.join(data_dir, 'train.csv'),
        historic_window + 1, forecast_horizon, device)
    valid_set = CustomLoadDataset(
        os.path.join(data_dir, 'valid.csv'),
        historic_window + 1, forecast_horizon, device)

    # Create DataLoaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)

    # Configuring Model
    hidden_nodes = args.hidden_size
    input_size = 1
    output_size = 1

    n_iterations = args.num_epochs
    learning_rate = args.learning_rate

    trafo_model = True

    if trafo_model:
        model = LoadForecasterTransformer(num_tokens=historic_window, dim_model=32, num_heads=8,
                                          num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.0)
    else:
        model = LoadForecaster(input_size, hidden_nodes, output_size, device=device)

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = torch.zeros(n_iterations)
    val_loss = torch.zeros(n_iterations)

    for epoch in range(n_iterations):
        # training phase
        model.train()
        loader = train_loader

        for input_seq, target_seq in loader:
            # Now we shift the tgt by one so  we predict the token at pos 1
            y_input = target_seq[:, :-1]
            y_expected = target_seq[:, 1:]

            # Get mask to mask out the next words
            sequence_length = target_seq.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)
            pred = model(input_seq, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)
            loss = criterion(predict, target_seq)
        
            train_loss[epoch] += loss.item()

            model.zero_grad()
            loss.backward()
            optim.step()

        train_loss[epoch] /= len(loader)

        # validation phase
        model.eval()
        loader = valid_loader
        for input_seq, target_seq in loader:
            with torch.no_grad():
                hidden = model.init_hidden(batch_size)
                predict, hidden = model(input_seq, hidden)

                loss = criterion(predict, target_seq)
                val_loss[epoch] += loss.item()

        val_loss[epoch] /= len(loader)
        print(f"Epoch {epoch + 1}: Training Loss = {train_loss[epoch]}, Validation Loss = {val_loss[epoch]}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_file = os.path.join(args.save_dir, "energy_baseline.pt")
        torch.save(model.state_dict(), save_file)
        print(f"Done! Saved model weights at {save_file}")


if __name__ == '__main__':
    main()
