import matplotlib.pyplot as plt
import seaborn as sns

def make_slice_plots(eta_true, writer, i, title): 
    res_xy = eta_true.shape[0]
    eta_slice = eta_true.reshape(res_xy, res_xy, 16, -1).mean(3)
    fig, axes = plt.subplots(4, 4, figsize=(18., 15.))
    for j in range(16): 
        k = j % 4
        l = j // 4
        ax = axes[l, k]
        ax.set_xlim([res_xy-1, 0])
        ax.set_ylim([0, res_xy-1])
        ax.set_title('Eta Field Slice %d'%j)
        heatmap_im = ax.imshow(eta_slice[:, :, j].T, vmin=1., vmax=eta_true.max())
        plt.colorbar(heatmap_im, ax=ax)
    writer.add_figure(title, fig, i)
    plt.close()