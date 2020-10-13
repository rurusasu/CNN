import torch
import numpy as np


def visualize_bounding_box(rgb, corners_pred, corners_targets=None,
                           centers_pred=None, centers_targets=None,
                           save=False, save_fn=None):
    """
    """

    # 型判定を行う。
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.permute(0, 2, 3, 1).detach().cpu().numpy()
    rgb = rgb.astype(nu.uint8)

    batch_size = corners_pred.shape[0]
    for idx in range(batch_size):
        _, ax = plt.subplots(1)
        ax.imshow(rgb[idx])
        ax.add_patch(
            patches.Polygon(xy=corners_pred[idx, 0][[
                            0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b')
        )
        ax.add_patch(
            pathes.Polygon(xy=corners_pred[idx, 0][[
                           5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=False, edgecolor='b')
        )
        if corners_targets is not None:
            ax.add_patch(pathes.Polygon(xy=corners_targets[idx, 0][[
                         0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
            ax.add_patch(patches.Polygon(xy=corners_targets[idx, 0][[
                         5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        if centers_pred is not None:
            ax.plot(centers_pred[idx, 0, 0], centers_pred[idx, 0, 1], '*')
        if centers_targets is not None:
            ax.plot(centers_targets[idx, 0, 0], centers_pred[idx, 0, 1], '*')
        if not save:
            plt.show()
        else:
            plt.savefig(save_fn.format(idx))
        plt.close()
