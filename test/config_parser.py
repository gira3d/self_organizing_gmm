import os
import configargparse


class ConfigParser(configargparse.ArgParser):
    def __init__(self):
        super().__init__(default_config_files=[
            os.path.join(os.path.dirname(__file__), 'default_test_config.yaml')
        ], conflict_handler='resolve')

        # for 4D point cloud dataset
        self.add_argument('--path_dataset', type=str,
                          help='Path to the dataset folder.')
        self.add_argument('--dataset', type=str,
                          help='Name of the dataset.')
        self.add_argument('--frames', type=int, nargs='+',
                          help='Frame used for testing.')

        # for 2D dataset
        self.add_argument('--n_samples', type=int,
                          help='Number of samples to generate random dataset.')
        self.add_argument('--n_components', type=int,
                          help='Number of components/blobs in the dataset.')

        # GBMS
        self.add_argument('--bandwidth', type=float, help='Bandwidth for GBMS.')

        # General
        self.add_argument('--show_plots', dest='show_plots', default=False, action='store_true',
                          help='Show plots during tests.')


    def get_config(self):
        config = self.parse_args()
        return config
