import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rnncode.load_dataset import load_data
from rnncode.stock_dataset import StockDataset
from rnncode.rnn import RNN
from rnncode.gru import GRU
from rnncode.lstm import LSTM
from rnncode.train_test import train_model,test_model
from rnncode.plot import plot_results
# 1= RNN, 2 = GRU, 3 = LSTM
MODEL_TO_USE = 2
USE_PRETRAINED = False 

def main():
    torch.manual_seed(42)
    csv_file = 'data\Google_Stock_Price_Train.csv'
    test_csv_file = 'data\Google_Stock_Price_Train.csv'

    prev_len = 3
    batch_size = 32
    hidden_size = 64
    num_layers = 3
    epochs = 500
    learning_rate = 0.001
    params = {}
    params["prev_len"] = prev_len
    params["batch_size"] = batch_size 
    params["hidden_size"] =hidden_size
    params["num_layers"] = num_layers 
    params["epochs"] = epochs
    params["learning_rate"] = learning_rate

    train_data, scaler = load_data(csv_file,True)
    train_dataset = StockDataset(train_data, prev_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # If CUDA is available use that, else run on the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use switch variable to choose which model to use
    if MODEL_TO_USE == 1:
        params["model"] = "RNN"
        model = RNN(input_size=train_data.shape[1], hidden_size=hidden_size, num_layers=num_layers).to(device)
    elif MODEL_TO_USE == 2:
        params["model"] = "GRU"
        model = GRU(input_size=train_data.shape[1], hidden_size=hidden_size, num_layers=num_layers, output_size=train_data.shape[1]).to(device)
    elif MODEL_TO_USE == 3: 
        params["model"] = "LSTM"       
        model = LSTM(input_size=train_data.shape[1], hidden_size=hidden_size, num_layers=num_layers, output_size=train_data.shape[1]).to(device)

    # If we want to train the model
    if not USE_PRETRAINED:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_model(model, train_loader, criterion, optimizer, device, epochs, params)
        # Save the model to file
        torch.save(model.state_dict(), f"{params['model']}_{params['epochs']}.par")
    else:
        # Use a pretrained model
        model.load_state_dict(torch.load(f"{params['model']}_{params['epochs']}.par", weights_only=True))
    

    test_data, scaler = load_data(test_csv_file,False,0.7,False)

    # Test the model
    test_predictions, test_actuals = test_model(
        model, test_data, prev_len, device, scaler,params
    )

    # Set up plot output file names depending on model used
    plot_name = "data/"
    if MODEL_TO_USE == 1:
        plot_name = f"{plot_name}RNN"
    elif MODEL_TO_USE == 2:
        plot_name = f"{plot_name}GRU"
    elif MODEL_TO_USE == 3:
        plot_name = f"{plot_name}LSTM"

    # Generate plots
    plot_results(test_actuals=test_actuals,test_predictions=test_predictions,plot_name = plot_name)
if __name__ == "__main__":
    main()
