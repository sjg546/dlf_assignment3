import torch
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, dataloader, criterion, optimizer, device, epochs, params):
    loss_to_plot = []
    epochs_to_drop = 20
    # Set model to training mode
    model.train()
    # Iterate epoch number of times
    for epoch in range(epochs):
        # Reset loss at the start of each epoch
        total_loss = 0
        for inputs, targets in dataloader:
            # Train model using back propogation
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        loss_to_plot.append(epoch_loss)
        print(f"{epoch+1}/{epochs}: Loss =  {epoch_loss}")
    

    plt.plot()
    plt.plot(loss_to_plot[epochs_to_drop:])
    plt.ylabel("Epoch Loss")
    plt.xlabel("Epoch")

    plt.title(f"{params['model']} - Epoch {params['epochs']} Loss")
    plt.savefig(f"{params['model']}-Epoch{params['epochs']}_loss.png")

    
def test_model(model, test_data, seq_len, device, scaler,params):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for i in range(len(test_data) - seq_len - 1):
            inputs = test_data[i:i + seq_len]
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(device)
            
            pred = model(inputs).squeeze(0).cpu().numpy()
            
            actual = test_data[i + seq_len:i + seq_len + 1]

            # Append predictions and actuals
            predictions.append(pred)
            actuals.append(actual)
    
    # convert predictions back to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, test_data.shape[1]))
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, test_data.shape[1]))
    

    # Metric calculations
    rmse = {}
    mape = {}
    rmse["Open"] = float(np.sqrt(np.mean((predictions[0:,] - actuals[0:,]) ** 2)))
    rmse["High"] =  float(np.sqrt(np.mean((predictions[1:,] - actuals[1:,]) ** 2)))
    rmse["Low"] =   float(np.sqrt(np.mean((predictions[2:,] - actuals[2:,]) ** 2)))
    rmse["Close"] = float(np.sqrt(np.mean((predictions[3:,] - actuals[3:,]) ** 2)))
    rmse["Total"] = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
    
    mape["Open"] = float(np.mean(np.abs(((predictions[0:,] -  actuals[0:,])/actuals[0:,]) * 100)))
    mape["High"] =  float(np.mean(np.abs(((predictions[1:,] - actuals[1:,])/actuals[1:,]) * 100)))
    mape["Low"] =   float(np.mean(np.abs(((predictions[2:,] - actuals[2:,])/actuals[2:,]) * 100)))
    mape["Close"] = float(np.mean(np.abs(((predictions[3:,] - actuals[3:,])/actuals[3:,]) * 100)))
    mape["Total"] = float(np.mean(np.abs((actuals - predictions) / actuals)) * 100)

    with open('results.txt', 'a') as f:
        print(f"{params}:{rmse}",file=f)
    # Formatted for simple entry into latex table
    print(f"{params['model']} & {rmse['Open']:.3f}& {rmse['High']:.3f}& {rmse['Low']:.3f}& {rmse['Close']:.3f}& {rmse['Total']:.3f} \\\\")
    print(f"{params['model']} & {mape['Open']:.3f}& {mape['High']:.3f}& {mape['Low']:.3f}& {mape['Close']:.3f}& {mape['Total']:.3f} \\\\")
    
    print(f"RMSE = {rmse}")
    print(f"MAPE = {mape}")
  
    return predictions, actuals
