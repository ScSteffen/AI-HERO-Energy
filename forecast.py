import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from pandas import DataFrame
from dataset import AllCitiesDataset

# TODO: import your model
from model import LoadForecaster as SubmittedModel

import numpy as np


def forecast(forecast_model, forecast_set, device):
    forecast_model.to(device)
    forecast_model.eval()

    # batch_size = 1
    # forecast_loader = DataLoader(forecast_set, batch_size=batch_size, shuffle=False)
    # forecasts = torch.zeros([len(forecast_set), 1], device=device)
    # batch x zeit x features
    input = forecast_set.dataset
    pred_list = []
    with torch.no_grad():
        hidden = forecast_model.init_hidden(input.size()[0])
        prediction, hidden = forecast_model(input, hidden)

    out = np.zeros((input.size()[0] * input.size()[1], 1))
    for i in range(prediction.size()[0]):
        # get scalings of this city
        city = forecast_set.index_to_city[i]
        scaler = forecast_set.scaling_dict[city]
        out[i * input.size()[1]:(i + 1) * input.size()[1]] = prediction[i] * (scaler[1] - scaler[0]) + scaler[0]

    # rearrange to 168xnumWeeks matrix
    out2 = np.reshape(out, (-1, 168))
    return out2


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--weights_path", type=str,
                        default='/hkfs/work/workspace/scratch/bh6321-energy_challenge/AI-HERO/energy_baseline.pt',
                        help="Model weights path")  # TODO: adapt to your model weights path
    parser.add_argument("--save_dir", type=str, help='Directory where weights and results are saved', default='.')
    parser.add_argument("--data_dir", type=str, help='Directory containing the data you want to predict',
                        default='/hkfs/work/workspace/scratch/bh6321-energy_challenge/data')
    args = parser.parse_args()

    save_dir = args.save_dir
    data_dir = args.data_dir

    weights_path = args.weights_path

    # load model with pretrained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TODO: adjust arguments according to your model
    model = SubmittedModel(input_size=4, hidden_size=48, output_size=1, num_layer=1, device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # dataloader
    test_file = os.path.join(data_dir, 'test.csv')
    valid_file = os.path.join(data_dir, 'valid.csv')
    data_file = test_file if os.path.exists(test_file) else valid_file
    testset = AllCitiesDataset(data_file, 7 * 24, 7 * 24, device=device, test=True, data_dir=save_dir)

    # run inference
    forecasts = forecast(model, testset, device)

    df = DataFrame(forecasts)

    # save to csv
    result_path = os.path.join(save_dir, 'forecasts.csv')
    df.to_csv(result_path, header=False, index=False)

    print(f"Done! The result is saved in {result_path}")
