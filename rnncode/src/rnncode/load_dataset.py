import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_file, train=True, train_pct=0.7,standalone=False):
    data = pd.read_csv(csv_file)
    if not standalone:
        if train:
            data = data.head(int(len(data)*train_pct))
        else:
            data = data.tail(int(len(data)*(1.0-train_pct)))
    data = data[['Open', 'High', 'Low', 'Close']]

    # Find fields that are strings and fix them up
    string_mask = data["Close"].apply(lambda x: isinstance(x, str))
    contains_string = string_mask.any().any()
    if contains_string:
        data['Close'] = data['Close'].str.replace("\"","")
        data['Close'] = data['Close'].str.replace(",","")

    # Convert all fields to numeric
    data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric)

    # Fix up invalid close entries
    data['Close'] = data.apply(lambda x: x['Close']/2 if x['Close'] > x['High'] else x['Close'], axis=1)

    # scale the data using MinMaxScalaer, need to return the scaler as its used in
    # the inverse transform during the test phase
    scaler = MinMaxScaler()
    #Apply the scaler
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler
