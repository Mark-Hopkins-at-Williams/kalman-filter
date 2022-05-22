from numpy.random import normal, multivariate_normal
from numpy import array, matmul, identity
from numpy.linalg import inv
from plane import CartesianPlane, AnimatedSprite, Slider
import pygame as pg


class KalmanFilter:
    def __init__(self, F, H, mu_0, sigma_0, sigma_x, sigma_z):
        self.F = F
        self.H = H
        self.mu_t = mu_0
        self.sigma_t = sigma_0
        self.sigma_x = sigma_x
        self.sigma_z = sigma_z

    def kalman_gain_matrix(self):
        ftf = matmul(matmul(self.F, self.sigma_t), self.F.T)
        inverse_term = inv(matmul(matmul(self.H, ftf + self.sigma_x), self.H.T) + self.sigma_z)
        gain = matmul(matmul(ftf + self.sigma_x, self.H.T), inverse_term)
        return gain

    def update(self, z_1):
        kalman_gain = self.kalman_gain_matrix()
        old_mu_t = self.mu_t
        f_times_mu_t = matmul(self.F, self.mu_t)
        self.mu_t = f_times_mu_t + matmul(kalman_gain, z_1 - matmul(self.H, f_times_mu_t))
        self.sigma_t = matmul(identity(len(z_1)) - matmul(kalman_gain, self.H),
                              matmul(matmul(self.F, self.sigma_t), self.F.T) + self.sigma_x)
        return tuple(self.mu_t - old_mu_t)


class Squid(AnimatedSprite):

    SQUID_CELLS = []
    for i in range(1, 6):
        SQUID_CELLS += [f"images/squid{i}.png"] * 3
    for i in range(4, 1, -1):
        SQUID_CELLS += [f"images/squid{i}.png"] * 3

    def __init__(self, initial_x, initial_y):
        super().__init__((initial_x, initial_y), Squid.SQUID_CELLS, cell_scale=0.09)


class Tracker(AnimatedSprite):

    TRACKER_CELLS = []
    for i in range(1, 9):
        TRACKER_CELLS += [f"images/tracker{i}.png"] * 3
    for i in range(7, 1, -1):
        TRACKER_CELLS += [f"images/tracker{i}.png"] * 3

    def __init__(self, initial_x, initial_y):
        super().__init__((initial_x, initial_y),
                         Tracker.TRACKER_CELLS, cell_scale=0.09)


class TransitionModel:
    def __init__(self, F, sigma_x):
        self.F = array(F)
        self.sigma_x = array(sigma_x)

    def next_location(self, x_t, y_t):
        location = array([x_t, y_t]).T
        next_loc = multivariate_normal(matmul(self.F, location), self.sigma_x)
        return tuple(next_loc)


class Sensor:
    def __init__(self, H, sigma_z):
        self.H = array(H)
        self.sigma_z = array(sigma_z)

    def sense(self, x_t, y_t):
        location = array([x_t, y_t]).T
        sensed = multivariate_normal(matmul(self.H, location), self.sigma_z)
        return tuple(sensed)


class SquidHunt:

    def __init__(self):
        pg.init()
        if not pg.font:
            print("Warning, fonts disabled")
        if not pg.mixer:
            print("Warning, sound disabled")
        else:
            pg.mixer.init()
            pg.mixer.music.load('sounds/sonar.wav')
            pg.mixer.music.play(-1)
        pg.display.set_caption("Sonar")
        pg.mouse.set_visible(True)
        self.plane = CartesianPlane(x_max=30, y_max=10, screen_width=1200, screen_height=400)
        self.skittishness = Slider(20, 20, 100, "skittishness", initial_percentage=0.1)
        self.plane.add_widget(self.skittishness)
        self.sensor_noise = Slider(20, 80, 100, "sensor noise", initial_percentage=0.1)
        self.plane.add_widget(self.sensor_noise)
        self.position = (15, 5)
        self.squid = Squid(self.position[0], self.position[1])
        self.tracker = Tracker(self.position[0], self.position[1])
        self.plane.add_sprite(self.squid)
        self.plane.add_sprite(self.tracker)

    def read_sliders(self):
        new_sigma_x = array([[2 * self.skittishness.current_percentage(), 0],
                             [0, 2 * self.skittishness.current_percentage()]])
        new_sigma_z = array([[4 * self.sensor_noise.current_percentage(), 0],
                             [0, 4 * self.sensor_noise.current_percentage()]])
        return new_sigma_x, new_sigma_z

    def start(self):
        sigma_x, sigma_z = self.read_sliders()
        kalman = KalmanFilter(identity(2), identity(2),
                                   mu_0=array(list(self.position)).T,
                                   sigma_0=array([[0.3, 0], [0, 0.4]]),
                                   sigma_x=sigma_x,
                                   sigma_z=sigma_z)
        sensor = Sensor(identity(2), sigma_z)
        transition_model = TransitionModel(identity(2), sigma_x)
        clock = pg.time.Clock()
        going = True
        while going:
            clock.tick(60)
            if self.squid.is_stationary():
                sigma_x, sigma_z = self.read_sliders()
                kalman.sigma_x = sigma_x
                kalman.sigma_z = sigma_z
                transition_model.sigma_x = sigma_x
                sensor.sigma_z = sigma_z
                delta_x, delta_y = transition_model.next_location(0, 0)
                valid_x, valid_y = self.plane.in_bounds(self.position[0] + delta_x,
                                                        self.position[1] + delta_y)
                if not valid_x:
                    delta_x = -delta_x
                if not valid_y:
                    delta_y = -delta_y
                self.squid.move(delta_x, delta_y)
                self.position = (self.position[0] + delta_x, self.position[1] + delta_y)
                observation = sensor.sense(self.position[0], self.position[1])
                kalman_delta_x, kalman_delta_y = kalman.update(observation)
                self.tracker.move(kalman_delta_x, kalman_delta_y)
            for event in pg.event.get():
                self.plane.notify(event)
                if event.type == pg.QUIT:
                    going = False
            self.plane.refresh()
        pg.quit()


def main():
    hunt = SquidHunt()
    hunt.start()


if __name__ == '__main__':
    main()
