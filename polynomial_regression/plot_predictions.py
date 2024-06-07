import matplotlib.animation as animation, numpy as np, pandas as pd, matplotlib.pyplot as plt

def update_plot(pred_matrix, argsort, x1_extended_, y, line2, ax, scat):
    def update(frame):
        # for each frame, update the data stored on each artist.
        y = pred_matrix[argsort, frame]
        # update the scatter plot:
        ax.set_title("Epoch " + str(frame * 50000))
        data = np.stack([x1_extended_, y]).T
        scat.set_offsets(data)
        # update the line plot:
        return (scat, line2)

    return update


def plot_pred_matrix(pred_matrix, x1, y, x1_extended, y_extended, bias_constraint = True):
    tmp_df = pd.DataFrame({"col": ["blue"] * pred_matrix.shape[0]})
    x1_min, x1_max = x1.min(), x1.max()
    argsort = np.argsort(x1_extended)
    x1_extended_ = x1_extended[argsort]
    y_extended_ = y_extended[argsort]
    tmp_df['x1'] = x1_extended_
    tmp_df['y'] = y_extended_
    tmp_df["col"][(x1_extended_ < x1_min) | (x1_extended_ > x1_max)] = "red"

    # fig, ax = plt.subplots()
    # mn, mx = pred_matrix.min() - 1, pred_matrix.max() + 1
    # artists = []
    # for i in range(pred_matrix.shape[1]):
    # 	tmp_preds = pred_matrix[argsort, i]
    # 	tmp_df['y_pred'] = tmp_preds
    # 	ax.set_title("Epoch " + str(i * 50000 + 1))
    # 	container = [ax.scatter(tmp_df['x1'], tmp_df['y_pred'], color = tmp_df['col'])]
    # 	artists.append(container)

    # ani = animation.ArtistAnimation(fig = fig, artists = artists, interval = 500)
    # plt.show()

    fig, ax = plt.subplots()
    scat = ax.scatter(x1_extended_, pred_matrix[argsort, 0], c = tmp_df["col"], s = 2, label = 'Predicted')
    line2 = ax.plot(x1_extended_, y_extended_, label = 'Actual')[0]
    ax.legend()
    
    update = update_plot(pred_matrix, argsort, x1_extended_, y, line2, ax, scat)
    ani = animation.FuncAnimation(fig = fig, func = update, frames = 100, interval = 100)
    if bias_constraint:
        ani.save(filename = "bias_constrained_preds.gif", writer = "pillow")
    else:
        ani.save(filename = "bias_unconstrained_preds.gif", writer = "pillow")
    
    plt.show()
    return 
