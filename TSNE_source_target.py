from sklearn.manifold import TSNE
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

res_dir = './tsne/results_txt/'
img_dir = './tsne/imgs/'


parser = argparse.ArgumentParser(description='Sigsiam Evaluating!')

parser.add_argument('--initial_dims', default=None, type=int, help='initial_dims')
parser.add_argument('--results_txt', default=None, type=str, help='results_txt')

args = parser.parse_args()        


initial_dims = args.initial_dims
digits = np.loadtxt(res_dir + args.results_txt.split('_')[0] + '/' + args.results_txt.split('_')[-1] +'.txt')

digits_data = digits[:, 1:initial_dims + 1]

digits_target = digits[:, 0]
print("begin")
X_tsne = TSNE(n_components=2, random_state=1).fit_transform(digits_data)
print("Done TSNE")
# X_tsne = PCA(n_components=2).fit_transform(digits_data)
# print("Done PCA")
font = {"color": "darkred",
        "size": 13,
        "family": "serif"}

plt.style.use("default")
plt.figure(figsize=(10, 8.5))
# plt.axis("on")
plt.xticks([])
plt.yticks([])
ax = plt.gca()
ax.spines['top'].set_linewidth('2.0')
ax.spines['right'].set_linewidth('2.0')
ax.spines['left'].set_linewidth('2.0')
ax.spines['bottom'].set_linewidth('2.0')

# choose number and color to plot the TSNE.
color_list = ['b', 'r', 'y', 'g', 'cyan', 'tomato', 'gold', 'lime']
gap_point = 4
for i in range(len(color_list)):
        digits_target_temp = digits_target == i
        X_tsne_temp = X_tsne[digits_target_temp, :]
        if i < gap_point:  
            plt.scatter(X_tsne_temp[:, 0], X_tsne_temp[:, 1], c=color_list[i], alpha=0.6, marker='$' + str(i) + '$',
                    )
        else:
            plt.scatter(X_tsne_temp[:, 0], X_tsne_temp[:, 1], c=color_list[i], alpha=0.6, marker='$' + str(i-gap_point) + '$',
                    )

# choose color to plot the TSNE.
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits_target, alpha=0.6,
#             cmap=plt.cm.get_cmap('tab10'))                
# plt.title("t-SNE", fontdict=font)
# cbar = plt.colorbar(ticks=range(10))
# cbar.set_label(label='digit value', fontdict=font)
# plt.clim(-0.5, 9.5)

plt.tight_layout()
if not os.path.exists(img_dir + args.results_txt.split('_')[0] + '/'):
        os.makedirs(img_dir + args.results_txt.split('_')[0] + '/')

plt.savefig(img_dir + args.results_txt.split('_')[0] + '/' + args.results_txt.split('_')[-1] + '.svg', format='svg')
plt.savefig(img_dir + args.results_txt.split('_')[0] + '/' + args.results_txt.split('_')[-1] + '.png')
