import numpy as np
from cprint import cprint
import unittest
from config_parser import ConfigParser
import time

from sogmm_py.utils import matrix_to_tensor, o3d_to_np, calculate_depth_metrics, np_to_o3d
from sogmm_py.utils import read_log_trajectory, ImageUtils
from sogmm_py.vis_open3d import VisOpen3D

# Datasets
from sklearn import datasets

# GBMS
from sklearn.cluster import MeanShift as PythonMS
from sklearn.neighbors import NearestNeighbors
from mean_shift_py import MeanShift as CppMS

# KInit
from sklearn.cluster import kmeans_plusplus
from kinit_py import KInitf2CPU

# EM
from sklearn.mixture import GaussianMixture as PythonGMM
from gmm_py import GMMf2CPU

# Plotting/Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# Point cloud fit
import os
import open3d as o3d
from sogmm_cpu import SOGMMf4Host, SOGMMLearner, SOGMMInference

# GMR Python
from gmr import GMM, plot_error_ellipses


def make_ellipse(mean, covariance, color, ax):
    v, w = np.linalg.eigh(covariance)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    v = 3.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(
        mean, v[0], v[1], 180 + angle, color=color
    )
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.4)
    ax.add_artist(ell)
    ax.scatter(mean[0], mean[1], s=40, color=color, edgecolors='black')
    ax.set_aspect("equal")


def make_ellipses_gmm(gmm, ax, case='py', color='red'):
    for n, mean in enumerate(gmm.means_):
        if case == 'py':
            covariances = gmm.covariances_[n][:2, :2]
            make_ellipse(mean, covariances, color, ax)
        else:
            covariances = matrix_to_tensor(gmm.covariances_, 2)[n][:2, :2]
            make_ellipse(mean, covariances, color, ax)


class ScikitPyImpl:
    '''Baseline Scikit impl'''

    def __init__(self, b):
        # GBMS
        self.gbms = PythonMS(bandwidth=b, bin_seeding=True)
        self.gbms_labels = None
        self.gbms_modes = None

        # KInit
        self.kinit_centers = None
        self.kinit_indices = None
        self.kinit_resp = None

        # EM
        self.gmm = PythonGMM(covariance_type="full", init_params="k-means++")

    def fit_gbms(self, X):
        self.gbms.fit(X)
        self.gbms_labels = self.gbms.labels_
        self.gbms_modes = len(np.unique(self.gbms_labels))

    def plot_gbms(self, X, ax):
        for k in range(self.gbms_modes):
            my_members = self.gbms_labels == k
            ax.scatter(X[my_members, 0], X[my_members, 1],
                       s=40, color=np.random.rand(3,), edgecolors='black')
        ax.set_title('ScikitPyImpl GBMS Output')

    def fit_kinit(self, X, K=None, rs=None):
        if K is None:
            K = self.gbms_modes

        N = X.shape[0]
        self.kinit_centers, self.kinit_indices = kmeans_plusplus(
            X, K, random_state=rs)

        self.kinit_resp = np.zeros((N, K))
        self.kinit_resp[self.kinit_indices, np.arange(K)] = 1

    def fit_em(self, X, K=None):
        if K is None:
            K = self.gbms_modes

        self.gmm = PythonGMM(n_components=K,
                             init_params='k-means++')
        self.gmm.fit(X)

    def plot_em(self, ax, c='red'):
        make_ellipses_gmm(self.gmm, ax, case='py', color=c)

    def fit(self, X):
        pass

    def plot_sogmm(self, ax, c='red'):
        pass


class CppImpl:
    '''CPP implementation'''

    def __init__(self, b):
        # GBMS
        self.gbms = CppMS(b)
        self.gbms_labels = None
        self.gbms_modes = None

        # KInit
        self.kinit_centers = None
        self.kinit_indices = None
        self.kinit_resp = None

        # EM
        self.gmm = GMMf2CPU()

        # Point cloud fit
        self.sogmm = SOGMMf4Host()
        self.learner = SOGMMLearner(b)
        self.inference = SOGMMInference()

    def fit_gbms(self, X):
        self.gbms.fit(X)
        self.gbms_modes = self.gbms.get_num_modes()

        cluster_centers = self.gbms.get_mode_centers()
        nbrs = NearestNeighbors(
            n_neighbors=1, algorithm='kd_tree').fit(cluster_centers)
        self.gbms_labels = np.zeros(X.shape[0], dtype=int)
        _, idxs = nbrs.kneighbors(X)
        self.gbms_labels = idxs.flatten()

    def plot_gbms(self, X, ax):
        for k in range(self.gbms_modes):
            my_members = self.gbms_labels == k
            ax.scatter(X[my_members, 0], X[my_members, 1],
                       s=40, color=np.random.rand(3,), edgecolors='black')
        ax.set_title('CppImpl GBMS Output')

    def fit_kinit(self, X, K=None):
        if K is None:
            K = self.gbms_modes

        N = X.shape[0]

        kinit = KInitf2CPU()
        self.kinit_centers, self.kinit_indices = kinit.resp_calc(
            X, K)

        self.kinit_resp = np.zeros((N, K))
        self.kinit_resp[self.kinit_indices, np.arange(K)] = 1

    def fit_em(self, X, K=None):
        if K is None:
            K = self.gbms_modes

        self.gmm = GMMf2CPU(K)
        self.gmm.fit(X, self.kinit_resp)

    def plot_em(self, ax, c='green'):
        make_ellipses_gmm(self.gmm, ax, case='cpp', color=c)

    def fit(self, X):
        pass

    def plot_sogmm(self, ax, c='green'):
        pass


class SOGMMCPUTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # configs
        parser = ConfigParser()
        cls.config = parser.get_config()
        cls.plot_show = cls.config.show_plots

        # datasets
        cls.data_2d_1, _ = datasets.make_blobs(
            n_samples=cls.config.n_samples, centers=cls.config.n_components, random_state=10)
        cls.data_2d_2, _ = datasets.make_blobs(
            n_samples=cls.config.n_samples, centers=cls.config.n_components, random_state=20)
        cls.pcld_1 = o3d.io.read_point_cloud(os.path.join(cls.config.path_dataset,
                                                          f"pcd_{cls.config.dataset}_{cls.config.frames[0]}_decimate_4_0.pcd"),
                                             format='pcd')
        cls.pcld_2 = o3d.io.read_point_cloud(os.path.join(cls.config.path_dataset,
                                                          f"pcd_{cls.config.dataset}_{cls.config.frames[1]}_decimate_4_0.pcd"),
                                             format='pcd')

        # poses
        traj_file = os.path.join(
            cls.config.path_dataset, f"{cls.config.dataset}-traj.log")
        traj = read_log_trajectory(traj_file)
        cls.pose_1 = traj[cls.config.frames[0]].pose
        cls.pose_2 = traj[cls.config.frames[1]].pose

        # utility
        K = np.eye(3)
        K[0, 0] = 525.0
        K[1, 1] = 525.0
        K[0, 2] = 319.5
        K[1, 2] = 239.5

        cls.iu = ImageUtils(K)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # start runners at the beginning of every test
        self.scikit = ScikitPyImpl(self.config.bandwidth)
        self.cpp = CppImpl(self.config.bandwidth)

    def tearDown(self):
        # deleter runners at the beginning of every test
        del self.scikit
        del self.cpp

    @unittest.skip("Skipping test_gbms")
    def test_gbms(self):
        self.gbms_fig, self.gbms_ax = plt.subplots(1, 2)

        self.scikit.fit_gbms(self.data_2d_1)
        self.scikit.plot_gbms(self.data_2d_1, self.gbms_ax[0])

        self.cpp.fit_gbms(self.data_2d_1)
        self.cpp.plot_gbms(self.data_2d_1, self.gbms_ax[1])

        self.assertEqual(self.scikit.gbms_modes, self.cpp.gbms_modes)

        if self.plot_show:
            self.gbms_ax[0].set_aspect("equal")
            self.gbms_ax[1].set_aspect("equal")
            self.gbms_fig.tight_layout()
            plt.show()

    @unittest.skip("Skipping test_em")
    def test_em(self):
        self.em_fig, self.em_ax = plt.subplots()

        self.scikit.fit_gbms(self.data_2d_1)
        self.scikit.fit_kinit(self.data_2d_1)
        self.scikit.fit_em(self.data_2d_1)
        self.scikit.plot_em(self.em_ax)

        self.cpp.fit_gbms(self.data_2d_1)
        self.cpp.fit_kinit(self.data_2d_1)
        self.cpp.fit_em(self.data_2d_1)
        self.cpp.plot_em(self.em_ax)

        if self.plot_show:
            self.em_ax.set_aspect("equal")
            self.em_fig.tight_layout()

    @unittest.skip("Skipping test_likelihood_scores")
    def test_likelihood_scores(self):
        pcld = o3d_to_np(self.pcld_1)
        d = np.array([np.linalg.norm(x) for x in pcld[:, 0:3]])[:, np.newaxis]
        g = pcld[:, 3][:, np.newaxis]
        Y = np.concatenate((d, g), axis=1)

        self.cpp.learner.fit(Y, pcld, self.cpp.sogmm)

        points = self.cpp.inference.generate_pcld_4d(self.cpp.sogmm,
                                                     300000,
                                                     3.0)
        f4, p4, r4, rm4, rmstd4 = calculate_depth_metrics(
            np_to_o3d(pcld[:, 0:3]), np_to_o3d(points))
        cprint.info(
            f"4D sampling: f-score {f4}, precision {p4}, recall {r4}, recon. mean {rm4}, recon. std. {rmstd4}")
        points3d = self.cpp.inference.generate_pcld_3d(self.cpp.sogmm,
                                                       300000,
                                                       3.0)
        f3, p3, r3, rm3, rmstd3 = calculate_depth_metrics(
            np_to_o3d(pcld[:, 0:3]), np_to_o3d(points3d))
        cprint.info(
            f"3Dpart sampling: f-score {f3}, precision {p3}, recall {r3}, recon. mean {rm3}, recon. std. {rmstd3}")

        pcld_new = o3d_to_np(self.pcld_2)
        scores_4d = self.cpp.inference.score_4d(pcld_new, self.cpp.sogmm)
        novel_pcld_1 = pcld_new[scores_4d < -np.pi, 0:3]
        scores_3d = self.cpp.inference.score_3d(
            pcld_new[:, 0:3], self.cpp.sogmm)
        novel_pcld_2 = pcld_new[scores_3d < -np.pi, 0:3]

        if self.plot_show:
            o3d.visualization.draw_geometries(
                [np_to_o3d(novel_pcld_1), np_to_o3d(novel_pcld_2, color=[1.0, 0.0, 0.0])])

    @unittest.skip("Skipping test_2d_conditional")
    def test_2d_conditional(self):
        self.cpp.fit_kinit(self.data_2d_1, K=4)
        self.cpp.fit_em(self.data_2d_1, K=4)
        gmm = GMM(n_components=4, priors=self.cpp.gmm.weights_,
                  means=self.cpp.gmm.means_, covariances=matrix_to_tensor(self.cpp.gmm.covariances_, 2))

        plt.figure(figsize=(10, 5))

        ax = plt.subplot(131)
        ax.set_title("Dataset and GMM")
        ax.scatter(self.data_2d_1[:, 0], self.data_2d_1[:, 1], s=1)
        colors = ["r", "g", "b", "orange"]
        plot_error_ellipses(ax, gmm, colors=colors)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax = plt.subplot(132)
        ax.set_title("Conditional Distribution p(y | x)")
        Y = np.linspace(-12, 8, 1000)
        Y_test = Y[:, np.newaxis]
        X_test = 0.5
        conditional_gmm = gmm.condition([0], [X_test])
        p_of_Y = conditional_gmm.to_probability_density(Y_test)
        ax.plot(Y, p_of_Y, color="k", label="GMR", lw=3)
        for component_idx in range(conditional_gmm.n_components):
            p_of_Y = (conditional_gmm.priors[component_idx]
                      * conditional_gmm.extract_mvn(
                component_idx).to_probability_density(Y_test))
            ax.plot(Y, p_of_Y, color=colors[component_idx],
                    label="Component %d" % (component_idx + 1))
        ax.set_xlabel("y")
        ax.set_ylabel("$p(y|x=%.1f)$" % X_test)
        ax.legend(loc="best")

        ax = plt.subplot(133)
        ax.set_title("Conditional Distribution p(x | y)")
        X = np.linspace(-12, 8, 1000)
        X_test = X[:, np.newaxis]
        Y_test = -10.0
        conditional_gmm = gmm.condition([1], [Y_test])
        p_of_X = conditional_gmm.to_probability_density(X_test)
        ax.plot(X, p_of_X, color="k", label="GMR", lw=3)
        for component_idx in range(conditional_gmm.n_components):
            p_of_X = (conditional_gmm.priors[component_idx]
                      * conditional_gmm.extract_mvn(
                component_idx).to_probability_density(X_test))
            ax.plot(X, p_of_X, color=colors[component_idx],
                    label="Component %d" % (component_idx + 1))
        ax.set_xlabel("x")
        ax.set_ylabel("$p(x|y=%.1f)$" % Y_test)
        ax.legend(loc="best")

        plt.tight_layout()
        plt.show()

    def test_4d_conditional(self):
        pcld = o3d_to_np(self.pcld_1)

        d = np.array([np.linalg.norm(x) for x in pcld[:, 0:3]])[:, np.newaxis]
        g = pcld[:, 3][:, np.newaxis]
        Y = np.concatenate((d, g), axis=1)

        self.cpp.learner.fit(Y, pcld, self.cpp.sogmm)

        ts = time.time()
        points4d = self.cpp.inference.reconstruct(self.cpp.sogmm,
                                                  300000,
                                                  2.2)
        te = time.time()
        cprint.info(f"time taken for reconstruction {(te - ts):.2f} seconds")

        vis = VisOpen3D(visible=True)
        vis.add_geometry(np_to_o3d(points4d))
        vis.update_view_point(extrinsic=np.linalg.inv(self.pose_1))
        vis.render()


if __name__ == "__main__":
    unittest.main()
