import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanFilterWrapper:
    def __init__(self, process_noise: float, measurement_noise: float):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self._process_noise = process_noise
        self.kf.x = np.array([0, 0, 0, 0])  # initial state (location and velocity)
        dt = 1.0  # initial time step (recomputed dynamically in predict)
        self.kf.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )  # state transition matrix
        self.kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )  # measurement function
        self.kf.P *= 1000.0  # initial covariance matrix
        self.kf.R = np.eye(2) * measurement_noise  # measurement noise

        # Adjust Q for smoother velocity estimates
        self.kf.Q = (
            np.array(
                [
                    [dt**4 / 4, 0, dt**3 / 2, 0],
                    [0, dt**4 / 4, 0, dt**3 / 2],
                    [dt**3 / 2, 0, dt**2, 0],
                    [0, dt**3 / 2, 0, dt**2],
                ]
            )
            * process_noise
        )

    def predict(self, dt: float = None):
        """Predict state forward. If dt provided, update transition and process noise matrices."""
        if dt is not None and dt > 0:
            self.kf.F[0, 2] = dt
            self.kf.F[1, 3] = dt
            # Recompute Q to match actual dt (was previously fixed at dt=1.0)
            self.kf.Q = (
                np.array(
                    [
                        [dt**4 / 4, 0, dt**3 / 2, 0],
                        [0, dt**4 / 4, 0, dt**3 / 2],
                        [dt**3 / 2, 0, dt**2, 0],
                        [0, dt**3 / 2, 0, dt**2],
                    ]
                )
                * self._process_noise
            )
        self.kf.predict()

    def update(self, location: np.ndarray) -> np.ndarray:
        self.kf.update(location)
        return self.kf.x[:2]  # smoothed location
