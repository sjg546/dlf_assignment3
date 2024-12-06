import matplotlib.pyplot as plt

#Helper function for plotting the output
def plot_results(test_actuals,test_predictions,plot_name):
    print(test_actuals[:, 1])
    #Open
    plt.cla()
    plt.plot()

    plt.plot(test_actuals[:, 0], label="Actual")
    plt.plot(test_predictions[:, 0], label="Predicted")
    plt.legend()
    plt.title("Predicted vs Actual Values Open")
    plt.savefig(plot_name + "_Open.png" )
    plt.cla()
    plt.plot()
    #High
    plt.plot(test_actuals[:, 1], label="Actual")
    plt.plot(test_predictions[:, 1], label="Predicted")
    plt.legend()
    plt.title("Predicted vs Actual Values High")
    plt.savefig(plot_name + "_High.png")
    plt.cla()
    plt.plot()
    #Low
    plt.plot(test_actuals[:, 2], label="Actual")
    plt.plot(test_predictions[:, 2], label="Predicted")
    plt.legend()
    plt.title("Predicted vs Actual Values Low")
    plt.savefig(plot_name + "_Low.png")
    plt.cla()
    plt.plot()
    #Close
    plt.plot(test_actuals[:, 3], label="Actual")
    plt.plot(test_predictions[:, 3], label="Predicted")
    plt.legend()
    plt.title("Predicted vs Actual Values Close")
    plt.savefig(plot_name + "_Close.png")
    plt.cla()
    plt.plot()

