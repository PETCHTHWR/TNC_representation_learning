import os
import pickle
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import math
from torch.utils import data
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from yellowbrick.cluster import SilhouetteVisualizer

def create_simulated_dataset(window_size=50, path='./data/simulated_data/', batch_size=100):
    if not os.listdir(path):
        raise ValueError('Data does not exist')
    x = pickle.load(open(os.path.join(path, 'x_train.pkl'), 'rb'))
    y = pickle.load(open(os.path.join(path, 'state_train.pkl'), 'rb'))
    x_test = pickle.load(open(os.path.join(path, 'x_test.pkl'), 'rb'))
    y_test = pickle.load(open(os.path.join(path, 'state_test.pkl'), 'rb'))

    n_train = int(0.8*len(x))
    n_valid = len(x) - n_train
    n_test = len(x_test)
    x_train, y_train = x[:n_train], y[:n_train]
    x_valid, y_valid = x[n_train:], y[n_train:]

    datasets = []
    for set in [(x_train, y_train, n_train), (x_test, y_test, n_test), (x_valid, y_valid, n_valid)]:
        T = set[0].shape[-1]
        windows = np.split(set[0][:, :, :window_size * (T // window_size)], (T // window_size), -1)
        windows = np.concatenate(windows, 0)
        labels = np.split(set[1][:, :window_size * (T // window_size)], (T // window_size), -1)
        labels = np.round(np.mean(np.concatenate(labels, 0), -1))
        datasets.append(data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))

    trainset, testset, validset = datasets[0], datasets[1], datasets[2]
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def track_encoding(sample, label, encoder, window_size, path, sliding_gap=5):
    T = sample.shape[-1]
    windows_label = []
    encodings = []
    device = 'cuda'
    encoder.to(device)
    encoder.eval()
    for t in range(window_size//2,T-window_size//2,sliding_gap):
        windows = sample[:, t-(window_size//2):t+(window_size//2)]
        windows_label.append((np.bincount(label[t-(window_size//2):t+(window_size//2)].astype(int)).argmax()))
        encodings.append(encoder(torch.Tensor(windows).unsqueeze(0).to(device)).view(-1,))
    for t in range(window_size//(2*sliding_gap)):
        # fix offset
        encodings.append(encodings[-1])
        encodings.insert(0, encodings[0])
    encodings = torch.stack(encodings, 0)

    if 'waveform' in path:
        f, axs = plt.subplots(3)
        f.set_figheight(12)
        f.set_figwidth(27)
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 7])
        axs[0] = plt.subplot(gs[0])
        axs[1] = plt.subplot(gs[1])
        axs[2] = plt.subplot(gs[2])
        sns.lineplot(x=np.arange(0,sample.shape[1]/250, 1./250), y=sample[0], ax=axs[0])
        sns.lineplot(x=np.arange(0,sample.shape[1]/250, 1./250), y=sample[1], ax=axs[1])
        axs[1].margins(x=0)
        axs[1].grid(False)
        axs[1].xaxis.set_tick_params(labelsize=22)
        axs[1].yaxis.set_tick_params(labelsize=22)

    else:
        f, axs = plt.subplots(2)  # , gridspec_kw={'height_ratios': [1, 2]})
        f.set_figheight(10)
        f.set_figwidth(27)
        for feat in range(min(sample.shape[0], 5)):
            sns.lineplot(x=np.arange(sample.shape[1]), y=sample[feat], ax=axs[0])

    axs[0].set_title('Time series Sample Trajectory', fontsize=30, fontweight='bold')
    axs[0].xaxis.set_tick_params(labelsize=22)
    axs[0].yaxis.set_tick_params(labelsize=22)
    axs[-1].xaxis.set_tick_params(labelsize=22)
    axs[-1].yaxis.set_tick_params(labelsize=22)
    axs[-1].set_ylabel('Encoding dimensions', fontsize=28)
    axs[0].margins(x=0)
    axs[0].grid(False)
    t_0 = 0
    if not 'waveform' in path:
        for t in range(1, label.shape[-1]):
            if label[t]==label[t-1]:
                continue
            else:
                axs[0].axvspan(t_0, min(t+1, label.shape[-1]-1), facecolor=['y', 'g', 'b', 'r', 'c', 'm'][int(label[t_0])], alpha=0.5)
                t_0 = t
        axs[0].axvspan(t_0, label.shape[-1]-1 , facecolor=['y', 'g', 'b', 'r'][int(label[t_0])], alpha=0.5)
    axs[-1].set_title('Encoding Trajectory', fontsize=30, fontweight='bold')
    sns.heatmap(encodings.detach().cpu().numpy().T, cbar=False, linewidth=0.5, ax=axs[-1], linewidths=0.05, xticklabels=False)
    f.tight_layout()


    # sns.heatmap(encodings.detach().cpu().numpy().T, linewidth=0.5)
    plt.savefig(os.path.join("./plots/%s" % path, "embedding_trajectory_hm.pdf"))

    # windows = np.split(sample[:, :window_size * (T // window_size)], (T // window_size), -1)
    # windows = torch.Tensor(np.stack(windows, 0)).to(encoder.device)
    # windows_label = np.split(label[:window_size * (T // window_size)], (T // window_size), -1)
    # windows_label = torch.Tensor(np.mean(np.stack(windows_label, 0), -1 ) ).to(encoder.device)
    # encoder.to(encoder.device)
    # encodings = encoder(windows)

    pca = PCA(n_components=2)
    embedding = pca.fit_transform(encodings.detach().cpu().numpy())
    d = {'f1':embedding[:,0], 'f2':embedding[:,1], 'time':np.arange(len(embedding))}#, 'label':windows_label}
    df = pd.DataFrame(data=d)
    fig, ax = plt.subplots()
    ax.set_title("Trajectory")
    # sns.jointplot(x="f1", y="f2", data=df, kind="kde", size='time', hue='label')
    sns.scatterplot(x="f1", y="f2", data=df, hue="time")
    plt.savefig(os.path.join("./plots/%s" % path, "embedding_trajectory.pdf"))


def track_encoding_ADSB(sample, traj, encoder, window_size, path, idx, sliding_gap=5):
    T = sample.shape[-1]
    encodings = []
    device = 'cuda'
    encoder.to(device)
    encoder.eval()
    for t in range(window_size//2,T-window_size//2,sliding_gap):
        windows = sample[:, t-(window_size//2):t+(window_size//2)]
        encodings.append(encoder(torch.Tensor(windows).unsqueeze(0).to(device)).view(-1,))
    for t in range(window_size//(2*sliding_gap)):
        # fix offset
        encodings.append(encodings[-1])
        encodings.insert(0, encodings[0])
    encodings = torch.stack(encodings, 0)

    f, axs = plt.subplots(3, figsize=(20, 20), gridspec_kw={'height_ratios': [2, 1, 1]})  # Increase the width and height

    # Plot the first subplot as a line plot
    axs[0].plot(traj[0, :], traj[1, :])  # Use line plot
    axs[0].set_xlabel('X', fontsize=22)
    axs[0].set_ylabel('Y', fontsize=22)
    axs[0].set_title('Airport Trajectory', fontsize=30, fontweight='bold')
    axs[0].tick_params(axis='both', labelsize=22)
    axs[0].set_aspect('equal')
    axs[0].set_xlim(-170000, 170000) # Set x-axis limits
    axs[0].set_ylim(-170000, 25000) # Set aspect ratio to make it square

    # Plot the second subplot as a line plot
    for feat in range(min(sample.shape[0], 5)):
        axs[1].plot(np.arange(sample.shape[1]), sample[feat])
    axs[1].set_title('Time series of Flight Path Cosine Vector', fontsize=30, fontweight='bold')
    axs[1].set_xlabel('Time', fontsize=22)
    axs[1].set_ylabel('Value', fontsize=22)
    axs[1].tick_params(axis='both', labelsize=22)
    axs[1].grid(False)
    axs[1].set_aspect('auto')

    # Plot the third subplot as a heatmap
    sns.heatmap(encodings.detach().cpu().numpy().T, cbar=False, linewidth=0.5, ax=axs[2], linewidths=0.05,
                xticklabels=False)
    axs[2].set_title('Encoding Trajectory', fontsize=30, fontweight='bold')
    axs[2].set_ylabel('Encoding dimensions', fontsize=28)
    axs[2].tick_params(axis='both', labelsize=22)
    axs[2].set_aspect('auto')

    f.tight_layout()

    # Save the figure
    plt.savefig(os.path.join("./plots/%s" % path, f"embedding_trajectory_hm_{idx}.png"))

    # windows = np.split(sample[:, :window_size * (T // window_size)], (T // window_size), -1)
    # windows = torch.Tensor(np.stack(windows, 0)).to(encoder.device)
    # windows_label = np.split(label[:window_size * (T // window_size)], (T // window_size), -1)
    # windows_label = torch.Tensor(np.mean(np.stack(windows_label, 0), -1 ) ).to(encoder.device)
    # encoder.to(encoder.device)
    # encodings = encoder(windows)

    pca = PCA(n_components=2)
    embedding = pca.fit_transform(encodings.detach().cpu().numpy())
    d = {'f1':embedding[:,0], 'f2':embedding[:,1], 'time':np.arange(len(embedding))}#, 'label':windows_label}
    df = pd.DataFrame(data=d)
    fig, ax = plt.subplots()
    ax.set_title("Trajectory")
    # sns.jointplot(x="f1", y="f2", data=df, kind="kde", size='time', hue='label')
    sns.scatterplot(x="f1", y="f2", data=df, hue="time")
    # Saving the plot
    plt.savefig(os.path.join("./plots/%s" % path, f"embedding_trajectory_{idx}.png"))


def encode_ADSB(sample, encoder, window_size, sliding_gap=5):
    T = sample.shape[-1]
    encodings = []
    device = 'cuda'
    encoder.to(device)
    encoder.eval()
    for t in range(window_size//2,T-window_size//2,sliding_gap):
        windows = sample[:, t-(window_size//2):t+(window_size//2)]
        encodings.append(encoder(torch.Tensor(windows).unsqueeze(0).to(device)).view(-1,))
    for t in range(window_size//(2*sliding_gap)):
        # fix offset
        encodings.append(encodings[-1])
        encodings.insert(0, encodings[0])
    encodings = torch.stack(encodings, 0)
    return encodings.detach().cpu().numpy().T

def compute_cluster_scores(data, range_n_clusters):
    best_n_clusters = 0
    best_score = -1
    dbi_scores = []
    best_dbi_score = np.inf
    best_n_clusters_dbi = None
    sil_scores = []

    for n_clusters in range_n_clusters:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_assignments = hierarchical.fit_predict(data)

        silhouette_avg = silhouette_score(data, cluster_assignments)
        sil_scores.append(silhouette_avg)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n_clusters = n_clusters

        dbi = davies_bouldin_score(data, cluster_assignments)
        dbi_scores.append(dbi)
        print("For n_clusters =", n_clusters, "The Davies-Bouldin index is :", dbi)
        if dbi < best_dbi_score:
            best_dbi_score = dbi
            best_n_clusters_dbi = n_clusters

    print(f"Best number of clusters = {best_n_clusters} with silhouette score = {best_score}")
    print(f"Best number of clusters = {best_n_clusters_dbi} with DBI = {best_dbi_score}")

    return best_n_clusters_dbi, sil_scores, dbi_scores


def plot_scores(range_n_clusters, sil_scores, dbi_scores, path, filename):
    fig, ax = plt.subplots(2, 1, figsize=(10, 15))

    ax[0].plot(range_n_clusters, sil_scores)
    ax[0].set_xlabel('Number of clusters')
    ax[0].set_ylabel('Silhouette score')
    ax[0].set_title('Silhouette scores for varying number of clusters')

    ax[1].plot(range_n_clusters, dbi_scores)
    ax[1].set_xlabel('Number of clusters')
    ax[1].set_ylabel('Davies-Bouldin Index')
    ax[1].set_title('DBI for varying number of clusters')

    plt.tight_layout()
    plt.savefig(os.path.join("./plots/%s" % path, filename))


def plot_silhouette_visualizer(best_model, data, path, filename):
    best_model.fit(data)
    labels = best_model.labels_
    silhouette_values = silhouette_samples(data, labels)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(data) + (len(set(labels)) + 1) * 10])

    y_lower = 10
    for i in range(len(set(labels))):
        ith_cluster_silhouette_values = silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / len(set(labels)))
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")
    ax.axvline(x=silhouette_score(data, labels), color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.savefig(os.path.join("./plots/%s" % path, filename))


def plot_TSNE(best_model, traj, enc_traj, path, tsne_filename, cluster_filename, single_cluster_filename, max_cutoff_range=150):
    # Fit the data and get cluster assignments
    cluster_assignments = best_model.fit_predict(enc_traj)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(enc_traj)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.get_cmap('rainbow', best_model.n_clusters)
    for i in range(best_model.n_clusters):
        ax.scatter(tsne_results[cluster_assignments == i, 0],
                   tsne_results[cluster_assignments == i, 1],
                   label=f'Cluster {i + 1}',
                   color=cmap(i))
    ax.legend()
    ax.set_title("t-SNE Plot")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    plt.savefig(os.path.join("./plots/%s" % path, tsne_filename))

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(best_model.n_clusters):
        cluster_trajectories = traj[cluster_assignments == i]
        for trajectory in cluster_trajectories:
            ax.plot(trajectory[0], trajectory[1], color=cmap(i), label=f'Cluster {i + 1}')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    circle = plt.Circle((0, 0), max_cutoff_range * 1000, color='r', fill=False)
    ax.add_patch(circle)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig(os.path.join("./plots/%s" % path, cluster_filename))

    n_clusters = best_model.n_clusters
    rows = int(math.ceil(math.sqrt(n_clusters)))
    cols = int(math.ceil(n_clusters / rows))

    fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))

    for i in range(n_clusters):
        row = i // cols
        col = i % cols
        ax = axs[row, col]

        cluster_trajectories = traj[cluster_assignments == i]
        for trajectory in cluster_trajectories:
            ax.plot(trajectory[0], trajectory[1], color=cmap(i))
        ax.set_title(f"Cluster {i + 1}")
        circle = plt.Circle((0, 0), max_cutoff_range * 1000, color='r', fill=False)
        ax.add_patch(circle)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join("./plots/%s" % path, f"{single_cluster_filename}.png"))
    plt.close(fig)  # Close the figure to free up memory


def plot_traj_TSNE(traj, path, max_cutoff_range=150):
    traj_reshaped = traj.reshape(traj.shape[0], -1)
    range_n_clusters = list(range(8, 40))

    best_n_clusters_dbi, sil_scores, dbi_scores = compute_cluster_scores(traj_reshaped, range_n_clusters)
    plot_scores(range_n_clusters, sil_scores, dbi_scores, path, "scores_traj.png")

    best_model = AgglomerativeClustering(n_clusters=best_n_clusters_dbi)
    plot_silhouette_visualizer(best_model, traj_reshaped, path, "silhouette_visualizer_traj.png")
    plot_TSNE(best_model, traj, traj_reshaped, path, "ori_tsne.png", "ori_traj_cluster.png", "ori_traj_cluster_single", max_cutoff_range=max_cutoff_range)


def plot_enc_TSNE(sample, traj, encoder, window_size, path, sliding_gap=5, max_cutoff_range=150):
    enc_traj = np.array([encode_ADSB(sample[i, :, :], encoder, window_size, sliding_gap=sliding_gap) for i in
                         range(sample.shape[0])]).reshape((sample.shape[0], -1))
    range_n_clusters = list(range(8, 40))

    best_n_clusters_dbi, sil_scores, dbi_scores = compute_cluster_scores(enc_traj, range_n_clusters)
    plot_scores(range_n_clusters, sil_scores, dbi_scores, path, "scores_enc.png")

    best_model = AgglomerativeClustering(n_clusters=best_n_clusters_dbi)
    plot_silhouette_visualizer(best_model, enc_traj, path, "silhouette_visualizer_enc.png")
    plot_TSNE(best_model, traj, enc_traj, path, "tsne.png", "enc_cluster.png", "enc_cluster_single", max_cutoff_range=max_cutoff_range)

def plot_distribution(x_test, y_test, encoder, window_size, path, device, title="", augment=4, cv=0):
    checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)
    n_test = len(x_test)
    inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * augment)
    windows = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    windows_state = [np.round(np.mean(y_test[i % n_test, ind:ind + window_size], axis=-1)) for i, ind in
                     enumerate(inds)]
    encodings = encoder(torch.Tensor(windows).to(device))

    tsne = TSNE(n_components=2)
    embedding = tsne.fit_transform(encodings.detach().cpu().numpy())
    # pca = PCA(n_components=2)
    # embedding = pca.fit_transform(encodings.detach().cpu().numpy())
    # original_embedding = PCA(n_components=2).fit_transform(windows.reshape((len(windows), -1)))
    original_embedding = TSNE(n_components=2).fit_transform(windows.reshape((len(windows), -1)))


    df_original = pd.DataFrame({"f1": original_embedding[:, 0], "f2": original_embedding[:, 1], "state": windows_state})
    df_encoding = pd.DataFrame({"f1": embedding[:, 0], "f2": embedding[:, 1], "state": windows_state})
    # df_encoding = pd.DataFrame({"f1": embedding[:, 0], "f2": embedding[:, 1], "state": y_test[np.arange(4*n_test)%n_test, inds]})


    # Save plots
    if not os.path.exists(os.path.join("./plots/%s"%path)):
        os.mkdir(os.path.join("./plots/%s"%path))
    # plt.figure()
    fig, ax = plt.subplots()
    ax.set_title("Origianl signals TSNE", fontweight="bold")
    # sns.jointplot(x="f1", y="f2", data=df_original, kind="kde", hue='state')
    sns.scatterplot(x="f1", y="f2", data=df_original, hue="state")
    plt.savefig(os.path.join("./plots/%s"%path, "signal_distribution.pdf"))

    fig, ax = plt.subplots()
    # plt.figure()
    ax.set_title("%s"%title, fontweight="bold", fontsize=18)
    if 'waveform' in path:
        sns.scatterplot(x="f1", y="f2", data=df_encoding, hue="state", palette="deep")
    else:
        sns.scatterplot(x="f1", y="f2", data=df_encoding, hue="state")
    # sns.jointplot(x="f1", y="f2", data=df_encoding, kind="kde", hue='state')
    plt.savefig(os.path.join("./plots/%s"%path, "encoding_distribution_%d.pdf"%cv))


def model_distribution(x_train, y_train, x_test, y_test, encoder, window_size, path, device):
    checkpoint = torch.load('./ckpt/%s/checkpoint.pth.tar'%path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    # n_train = len(x_train)
    n_test = len(x_test)
    augment = 100

    # inds = np.random.randint(0, x_train.shape[-1] - window_size, n_train * 20)
    # windows = np.array([x_train[int(i % n_train), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    # windows_label = [np.round(np.mean(y_train[i % n_train, ind:ind + window_size], axis=-1))
    #                  for i, ind in enumerate(inds)]
    inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * augment)
    x_window_test = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    y_window_test = np.array([np.round(np.mean(y_test[i % n_test, ind:ind + window_size], axis=-1)) for i, ind in
                     enumerate(inds)])
    train_count = []
    if 'waveform' in path:
        encoder.to('cpu')
        x_window_test = torch.Tensor(x_window_test)
    else:
        encoder.to(device)
        x_window_test = torch.Tensor(x_window_test).to(device)

    encodings_test = encoder(x_window_test).detach().cpu().numpy()

    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(encodings_test, y_window_test)
    # preds = neigh.predict(encodings_test)
    _, neigh_inds = neigh.kneighbors(encodings_test)
    neigh_ind_labels = [np.mean(y_window_test[ind]) for ind in (neigh_inds)]
    label_var = [(y_window_test[ind]==y_window_test[i]).sum() for i, ind in enumerate(neigh_inds)]
    dist = (label_var)/10


def confidence_ellipse(mean, cov, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, edgecolor='navy',
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def trend_decompose(x, filter_size):
    df = pd.DataFrame(data=x.T)
    df = df.rolling(filter_size, win_type='triang').sum()
    s = df.loc[:, 0]
    f, axs = plt.subplots(1)
    print(s[filter_size-1:].shape, x[0,:-filter_size+1].shape)
    axs.plot(s[filter_size-1:], c='red')
    axs.plot(x[0,:-filter_size+1], c='blue')
    plt.show()


# Function to rotate vectors
def rotate_3D(vector, angle_degrees_z, angle_degrees_x):
    theta_z = np.radians(angle_degrees_z)
    theta_x = np.radians(angle_degrees_x)
    rotation_matrix_z = np.array([[np.cos(theta_z), np.sin(theta_z), 0],
                                  [-np.sin(theta_z), np.cos(theta_z), 0],
                                  [0, 0, 1]])
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(theta_x), np.sin(theta_x)],
                                  [0, -np.sin(theta_x), np.cos(theta_x)]])
    rotated_vector = np.dot(rotation_matrix_z, vector)
    rotated_vector = np.dot(rotation_matrix_x, rotated_vector)
    return rotated_vector

def rotate_3D_cuda(vector, angle_degrees_z, angle_degrees_x):
    vector = cp.asarray(vector)
    theta_z = cp.radians(angle_degrees_z)
    theta_x = cp.radians(angle_degrees_x)
    cz = float(cp.cos(theta_z))
    sz = float(cp.sin(theta_z))
    cx = float(cp.cos(theta_x))
    sx = float(cp.sin(theta_x))
    rotation_matrix_z = cp.array([[cz, sz, 0],
                                  [-sz, cz, 0],
                                  [0, 0, 1]])
    rotation_matrix_x = cp.array([[1, 0, 0],
                                  [0, cx, sx],
                                  [0, -sx, cx]])
    rotated_vector = cp.dot(rotation_matrix_z, vector)
    rotated_vector = cp.dot(rotation_matrix_x, rotated_vector)
    return rotated_vector.get()

def augment_with_offset(vector, off_x, off_y, off_z):
    offset = torch.stack([off_x, off_y, off_z], dim=0).to(vector.device)
    augmented_vector = vector + offset
    return augmented_vector

def augment_with_rotation(vectors, angle_degrees_z, angle_degrees_x):
    # Convert the angles to radians
    theta_z = torch.deg2rad(angle_degrees_z)
    theta_x = torch.deg2rad(angle_degrees_x)

    # Calculate the sine and cosine values of the angles
    cz = torch.cos(theta_z)
    sz = torch.sin(theta_z)
    cx = torch.cos(theta_x)
    sx = torch.sin(theta_x)

    # Define the rotation matrices
    rotation_matrix_z = torch.tensor([[cz, -sz, 0],
                                      [sz, cz, 0],
                                      [0, 0, 1]], dtype=vectors.dtype, device=vectors.device)
    rotation_matrix_x = torch.tensor([[1, 0, 0],
                                      [0, cx, -sx],
                                      [0, sx, cx]], dtype=vectors.dtype, device=vectors.device)

    # Apply the rotation using matrix multiplication
    rotated_vectors = torch.matmul(rotation_matrix_z, vectors)
    rotated_vectors = torch.matmul(rotation_matrix_x, rotated_vectors)

    return rotated_vectors


def augment_sect_tensor(tensor, p_r = 0.5, p_theta = 0.5, p_z = 0.5,
                        r_bins = 10, theta_bins = 24, z_bins = 10):
    """
    Augment a tensor by adding a random offset to each axis of each (3, 100) window.
    The offset for each axis is independently randomly selected from the set (-1, 0, 1) with probabilities specified by `center_portion`.

    Parameters:
    tensor (torch.Tensor): A tensor of shape (N, 3, 100).
    center_portion (float): Portion to be assigned to the center value in the distribution.

    Returns:
    tensor (torch.Tensor): The augmented tensor.
    """

    # Define the set of values and their probabilities
    values_r = torch.tensor([-2 / r_bins, 0, 2 / r_bins], dtype=torch.float32, device=tensor.device)
    values_theta = torch.tensor([-2 / theta_bins, 0, 2 / theta_bins], dtype=torch.float32, device=tensor.device)
    values_z = torch.tensor([-2 / z_bins, 0, 2 / z_bins], dtype=torch.float32, device=tensor.device)

    side_portion_r = (1.0 - p_r) / 2
    probabilities_r = torch.tensor([side_portion_r, p_r, side_portion_r], dtype=torch.float32, device=tensor.device)

    side_portion_theta = (1.0 - p_theta) / 2
    probabilities_theta = torch.tensor([side_portion_theta, p_theta, side_portion_theta], dtype=torch.float32, device=tensor.device)

    side_portion_z = (1.0 - p_z) / 2
    probabilities_z = torch.tensor([side_portion_z, p_z, side_portion_z], dtype=torch.float32, device=tensor.device)

    # Calculate number of windows and axes in your tensor
    num_windows, num_axes, window_length = tensor.shape

    # Generate offsets for each axis of each window
    offsets_r = torch.multinomial(probabilities_r, num_samples=num_windows, replacement=True)
    offsets_theta = torch.multinomial(probabilities_theta, num_samples=num_windows, replacement=True)
    offsets_z = torch.multinomial(probabilities_z, num_samples=num_windows, replacement=True)

    # Map the offsets to the corresponding values
    offsets_r = values_r[offsets_r]
    offsets_theta = values_theta[offsets_theta]
    offsets_z = values_z[offsets_z]

    # Stack the offsets
    offsets = torch.stack([offsets_r, offsets_theta, offsets_z], dim=1)

    # Reshape and repeat the offsets to match the shape of the tensor
    offsets = offsets.view(num_windows, num_axes, 1).repeat(1, 1, window_length)

    # Add the offsets to the tensor
    tensor += offsets

    return tensor
