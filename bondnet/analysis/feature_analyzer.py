import abc
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib as mpl
import matplotlib.pyplot as plt
from bondnet.layer.readout import ConcatenateMeanMax
from bondnet.analysis import umap_plot
from bondnet.utils import to_path


class FeatureAggregator:
    def __init__(self, dataset):
        self.dataset = dataset

    def _stack_feature(self, ntype):
        feature = []
        for g, _, _ in self.dataset:
            feature.append(g.nodes[ntype].data["ft"])
        return np.concatenate(feature)

    def _stack_feature_and_label(self, ntype="bond"):
        """
        Stack feature (each feature is the bond feature and concatenated with the mean
        and max of the features of the atoms constituting the bond) and label whose
        corresponding label indicator is True, i.e. we have energy for the label.
        """
        features = []
        labels = []
        for g, lb, _ in self.dataset:
            # indices of bond that has energy
            indices = [int(i) for i, v in enumerate(lb["indicator"]) if v == 1]
            labels.append(lb["value"][indices])
            features.append(g.nodes[ntype].data["ft"][indices])

        features = np.concatenate(features)
        labels = np.concatenate(labels)

        return features, labels

    def _stack_bond_feature_plus_atom_feature_and_label(self):
        """
        Stack feature and label whose corresponding label indicator is True, i.e. we
        have energy for the label.
        """
        features = []
        labels = []

        for g, lb, _ in self.dataset:
            # indices of bond that has energy
            indices = [int(i) for i, v in enumerate(lb["indicator"]) if v == 1]
            labels.append(lb["value"][indices])

            all_feats = {
                "atom": g.nodes["atom"].data["ft"],
                "bond": g.nodes["bond"].data["ft"],
                "global": g.nodes["global"].data["ft"],
            }

            etypes = [("atom", "a2b", "bond")]
            layer = ConcatenateMeanMax(etypes)
            rst = layer(g, all_feats)
            bond_feats = rst["bond"][indices]
            features.append(bond_feats)

        features = np.concatenate(features)
        labels = np.concatenate(labels)

        return features, labels


class BaseAnalyzer(abc.ABC):
    """
    Base analyzer for features.

    Args:
        features (2D array): each row is the features of a data point.
        metadata (dict): metadata (e.g. label) of each data point, with string of key
            and 1D array (should have the same size as `features`) as list.
    """

    def __init__(self, features, metadata=None):
        self.features = features
        self.metadata = metadata
        self.embedding = None
        self.model = None

    @classmethod
    def from_csv(cls, feature_file, metadata_file, sep=","):
        """
        Read csv files. feature_file should have neither header and row
        index and metadata_file should have header but not row index.

        For example, see: http://projector.tensorflow.org
        """
        features = pd.read_csv(feature_file, sep=sep, header=None, index_col=None)
        features = features.to_numpy()
        metadata = pd.read_csv(metadata_file, sep=sep, header=0, index_col=None)
        metadata = metadata.to_dict(orient="list")
        metadata = {k: np.asarray(v) for k, v in metadata.items()}

        return cls(features, metadata)

    def write_embedding_to_csv(self, filename, sep=","):
        if self.embedding is not None:
            df = pd.DataFrame(
                {"1st_dim": self.embedding[:, 0], "2nd_dim": self.embedding[:, 1]}
            )
            df.to_csv(filename, sep=sep, header=True, index=False)
        else:
            raise RuntimeError("`embedding` is None, call `compute()` to get it first")

    @abc.abstractmethod
    def compute(self, **kwargs):
        raise NotImplementedError

    def plot(self, metadata_key_as_color="energy", filename="embedding.pdf"):
        plot_scatter([self.embedding], [self.metadata[metadata_key_as_color]], filename)

    def plot_via_umap_points(
        self,
        filename="embedding.pdf",
        metadata_key_as_color="prediction",
        categorical_color=False,
    ):
        if not categorical_color:
            umap_plot.points(
                to_path(filename),
                self.embedding,
                values=self.metadata[metadata_key_as_color],
                theme="viridis",
            )
        else:
            umap_plot.points(
                to_path(filename),
                self.embedding,
                labels=self.metadata[metadata_key_as_color],
                color_key_cmap="tab20",
            )

    def plot_via_umap_interactive(
        self,
        filename="embedding.html",
        metadata_key_as_color="species",
        categorical_color=False,
    ):
        if not categorical_color:
            umap_plot.interactive(
                to_path(filename),
                self.model,
                values=self.metadata[metadata_key_as_color],
                hover_data=pd.DataFrame(self.metadata),
                theme="viridis",
                point_size=5,
            )
        else:
            umap_plot.interactive(
                to_path(filename),
                self.model,
                labels=self.metadata[metadata_key_as_color],
                hover_data=pd.DataFrame(self.metadata),
                color_key_cmap="tab20",
                point_size=5,
            )


class PCAAnalyzer(BaseAnalyzer):
    """
    PCA analysis for features.
    """

    def compute(self, n_components=2, **kwargs):
        self.model = PCA(n_components=n_components, **kwargs)
        embedding = self.model.fit_transform(self.features)
        self.embedding = embedding

        return self.embedding


class UMAPAnalyzer(BaseAnalyzer):
    """
    UMAP analysis for features.
    """

    def compute(
        self,
        n_neighbors=100,
        n_components=2,
        min_dist=0.5,
        metric="euclidean",
        verbose=True,
        n_epochs=1000,
        **kwargs,
    ):
        self.model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            verbose=True,
            n_epochs=n_epochs,
            **kwargs,
        )
        embedding = self.model.fit_transform(self.features)
        self.embedding = embedding

        return self.embedding


class StdevThreshold(FeatureAggregator):
    """
    Analyzer to check the standard deviation of the features.

    Args:
        dataset: a :class:`bondnet.data.dataset.ElectrolyteData` dataset.
    """

    def compute(self, ntype, threshold=1e-8):
        """
        Compute the stdandard deviation of each feature and print a report.

        Args:
            ntype (str): the node type of the graph where the features are stored, e.g.
                `atom` and `bond`.

        Returns:
            list: indices of features whose stdev is smaller than the threshold.
        """

        data = self._stack_feature(ntype)
        self.stdevs = np.std(data, axis=0)
        self.means = np.mean(data, axis=0)

        if np.all(self.stdevs <= threshold):
            raise ValueError(
                "No feature meets the stdev threshold {:.5f}".format(threshold)
            )

        feature_name = self.dataset.feature_name[ntype]
        max_name_len = max([len(name) for name in feature_name])
        print("=" * 80)
        print("Node type:", ntype)
        print(
            "feature"
            + " " * max_name_len
            + "stdev    mean   less than threshlod ({})".format(threshold)
        )
        indices = []
        for i, (name, s, m) in enumerate(zip(feature_name, self.stdevs, self.means)):
            if s <= threshold:
                less_than = "yes"
                indices.append(i)
            else:
                less_than = "no"
            fmt = (
                "{:2d} ({})" + " " * (max_name_len - len(name)) + " {:.5f}  {:.5f}  {}"
            )
            print(fmt.format(i, name, s, m, less_than))
        print("=" * 80)

        return indices


class PearsonCorrelation(FeatureAggregator):
    """
    Analyzer to check the Pearson correlation coefficient between the features.
    Args:
        dataset: a :class:`bondnet.data.dataset.ElectrolyteData` dataset.
    """

    def compute(self, ntype, exclude=None):
        """
        Compute the Pearson correlation coefficient.

        Args:
            ntype (str): the node type of the graph where the features are stored, e.g.
                `atom` and `bond`.
            exclude (list, optional): indices of features to ignore. This is useful to
                exclude features with 0 stdandard deviation. If `None`, nothing to
                exclude. Defaults to None.
        Returns:
            2D array: correlation between features
        """
        data = self._stack_feature(ntype)
        if exclude is not None:
            data = np.delete(data, exclude, axis=1)
        corr = np.corrcoef(data, rowvar=False)
        return corr


def plot_scatter(features, labels, filename):
    """
    Scatter plot for features and use labels as color.

    Args:
        features (list of 2D array)
        labels (list of 1D array)
        filename (str)
    """
    # plot
    all_marker = ["o", "D", "x", "<", "+"]

    # x and y range
    xmin = min([min(d[:, 0]) for d in features])
    xmax = max([max(d[:, 0]) for d in features])
    ymin = min([min(d[:, 1]) for d in features])
    ymax = max([max(d[:, 1]) for d in features])
    del_x = 0.1 * (xmax - xmin)
    del_y = 0.1 * (ymax - ymin)
    xmin -= del_x
    xmax += del_x
    ymin -= del_y
    ymax += del_y

    # ensure different class has the same color range
    cmin = min([min(i) for i in labels])
    cmax = max([max(i) for i in labels])

    for i, (data, color) in enumerate(zip(features, labels)):
        fig, ax = plt.subplots()
        print("Feature array {} num data points {}".format(i, len(data)))
        X = data[:, 0]
        Y = data[:, 1]
        sc = ax.scatter(
            X,
            Y,
            marker=all_marker[i],
            c=color,
            vmin=cmin,
            vmax=cmax,
            cmap=mpl.cm.viridis,
            ec=None,
        )
        fig.colorbar(sc)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        filename = to_path(filename)
        filename = filename.parent.joinpath(filename.stem + f"_{i}" + filename.suffix)
        fig.savefig(filename, bbox_inches="tight")


def plot_heat_map(matrix, labels, filename="heat_map.pdf", cmap=mpl.cm.viridis):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)), minor=False)
    ax.set_yticks(np.arange(len(labels)), minor=False)
    # label them with the respective list entries
    ax.set_xticklabels(labels, minor=False)
    ax.set_yticklabels(labels, minor=False)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # colorbar
    plt.colorbar(im)

    fig.savefig(filename, bbox_inches="tight")
