import numpy as np

class KalmanFilter:

    def __init__(self, F, B, H, Q, R, x0, P0):

        self.F = F  # State transition matrix

        self.B = B  # Control matrix

        self.H = H  # Measurement matrix

        self.Q = Q  # Process noise covariance

        self.R = R  # Measurement noise covariance

        self.x = x0  # Initial state

        self.P = P0  # Initial covariance -> statistisches Maß wie sich zwei Variablen gemeinsam verändern und ob ein linearer Zusammenhang zwischen ihnen besteht

    def predict(self, u=0):

        # Predict state: x = F*x + B*u
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        
        # Predict covariance: P = F*P*F.T + Q
        self.P = np.dot(np.dot(self.F, self.P),self.F.T) + self.Q

        return self.x

    def update(self, z):

        y = z - np.dot(self.H, self.x)

        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)

        I = np.eye(self.P.shape[0])

        self.P = np.dot(I - np.dot(K, self.H), self.P)

        return self.x


#def setup_gps_imu_filter():    
    # F = np.array([[1, dt, 0.5*dt**2], # State: [position, velocity, acceleration]

    #               [0, 1, dt],

    #               [0, 0, 1]])

    # H_gps = np.array([[1, 0, 0]])      # GPS measures position

    # H_imu = np.array([[0, 0, 1]])      # IMU measures acceleration

    # R_gps = np.array([[10]])           # GPS noise (meters)

    # R_imu = np.array([[0.1]])          # IMU noise (m/s²)

    # return F, H_gps, H_imu, R_gps, R_imu


def sensor_fusion_step(kf, sensor_type, measurement):



    if sensor_type == 'gps':

        kf.H = H_gps

        kf.R = R_gps

    elif sensor_type == 'imu':

        kf.H = H_imu

        kf.R = R_imu

    else:
        raise ValueError("Unbekannter Sensor")

    z = np.array([[measurement]])

    return kf.update(z)


dt = 0.1

F = np.array([
    [1, dt, 0.5 * dt**2],
    [0, 1, dt],
    [0, 0, 1]
])

B = np.zeros((3, 1))

Q = np.array([
    [0.01, 0, 0],
    [0, 0.01, 0],
    [0, 0, 0.1]
])

H_gps = np.array([
    [1, 0, 0]
])

H_imu = np.array([
    [0, 0, 1]
])

R_gps = np.array([[9.0]])
R_imu = np.array([[0.04]])

x0 = np.array([
    [0.0],
    [0.0],
    [0.0]
])

P0 = np.eye(3) * 100


kf = KalmanFilter(
    F=F,
    B=B,
    H=H_gps,
    Q=Q,
    R=R_gps,
    x0=x0,
    P0=P0
)

kf.predict() # Always predict first

sensor_fusion_step(kf, "imu", 12.2)

print(kf.x)