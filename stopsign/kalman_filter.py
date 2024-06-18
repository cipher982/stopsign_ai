import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanFilterWrapper:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([0, 0, 0, 0])  # initial state (location and velocity)
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])  # state transition matrix
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # measurement function
        self.kf.P *= 1000.0  # covariance matrix
        self.kf.R = np.array([[5, 0], [0, 5]])  # measurement noise
        self.kf.Q = np.eye(4)  # process noise

    def predict(self):
        self.kf.predict()

    def update(self, location):
        self.kf.update(location)
        return self.kf.x[:2]  # smoothed location
