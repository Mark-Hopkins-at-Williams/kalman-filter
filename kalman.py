from numpy.random import normal
from plane import CartesianPlane, AnimatedSprite
import pygame as pg


class KalmanFilter:
    def __init__(self, mu_0, sigma_0, sigma_x, sigma_z):
        self.sq_sigma_x = sigma_x ** 2
        self.sq_sigma_z = sigma_z ** 2
        self.t = 0
        self.mu_t = mu_0
        self.sq_sigma_t = sigma_0 ** 2

    def update(self, z_1):
        self.t += 1
        mu_t_old = self.mu_t
        self.mu_t = (((self.sq_sigma_t + self.sq_sigma_x) * z_1 + self.sq_sigma_z * self.mu_t)
                     / (self.sq_sigma_t + self.sq_sigma_x + self.sq_sigma_z))
        self.sq_sigma_t = (((self.sq_sigma_t + self.sq_sigma_x) * self.sq_sigma_z)
                           / (self.sq_sigma_t + self.sq_sigma_x + self.sq_sigma_z))
        return self.mu_t - mu_t_old


class Whale(AnimatedSprite):
    def __init__(self, initial_x, initial_y):
        super().__init__((initial_x, initial_y),
                         ["images/whale{}.png".format(index) for index in range(1, 11)],
                         cell_scale=0.06)

class GhostWhale(AnimatedSprite):
    def __init__(self, initial_x, initial_y):
        super().__init__((initial_x, initial_y),
                         ["images/whaleghost{}.png".format(index) for index in range(1, 9)] + ["images/whale{}.png".format(index) for index in range(7, 1, -1)],
                         cell_scale=0.06)


class ErrorBar(AnimatedSprite):
    def __init__(self, initial_x, initial_y):
        super().__init__((initial_x, initial_y), ["images/error.png"], cell_scale=0.03)


def noisy_move(mu_x, sigma_x):
    return normal(loc=mu_x, scale=sigma_x)


def sense(x, sigma_z):
    return normal(loc=x, scale=sigma_z)


def main():
    if not pg.font:
        print("Warning, fonts disabled")
    if not pg.mixer:
        print("Warning, sound disabled")
    else:
        pg.mixer.init()
        pg.mixer.music.load('sounds/sonar.wav')
        pg.mixer.music.play(-1)
    pg.init()
    pg.display.set_caption("Sonar")
    pg.mouse.set_visible(True)
    plane = CartesianPlane(x_max=5, y_max=20, screen_width=400, screen_height=600)
    sigma_0 = 0.3
    sigma_x = 0.2
    sigma_z = 0.4
    ideal_delta_y = -1.0
    ideal_y = 18.0
    actual_y = normal(loc=ideal_y, scale=sigma_0)
    whale = Whale(1, actual_y)
    kalman_whale = GhostWhale(2, ideal_y)
    kalman_error = ErrorBar(2, 0)
    prior_whale = GhostWhale(3, ideal_y)
    prior_error = ErrorBar(3, 0)
    likelihood_whale = GhostWhale(4, ideal_y)
    likelihood_error = ErrorBar(4, 0)
    plane.add_sprite(whale)
    plane.add_sprite(kalman_whale)
    plane.add_sprite(kalman_error)
    plane.add_sprite(prior_whale)
    plane.add_sprite(prior_error)
    plane.add_sprite(likelihood_whale)
    plane.add_sprite(likelihood_error)
    clock = pg.time.Clock()
    going = True
    kalman = KalmanFilter(mu_0=0, sigma_0=sigma_0, sigma_x=sigma_x, sigma_z=sigma_z)
    while going:
        clock.tick(60)
        if whale.is_stationary() and whale.y >= 1:
            kalman_error.move(0, abs(whale.y - kalman_whale.y))
            prior_error.move(0, abs(whale.y - prior_whale.y))
            likelihood_error.move(0, abs(whale.y - likelihood_whale.y))
            delta_y = noisy_move(mu_x=ideal_delta_y, sigma_x=sigma_x)
            future_y = whale.y + delta_y
            ideal_y += ideal_delta_y
            whale.move(0, delta_y)
            observation = sense(future_y, sigma_z)
            kalman_step = kalman.update(observation - ideal_y)
            kalman_whale.move(0, kalman_step + ideal_delta_y)
            prior_whale.move(0, ideal_delta_y)
            likelihood_whale.move(0, observation - likelihood_whale.y)
        for event in pg.event.get():
            plane.notify(event)
            if event.type == pg.QUIT:
                going = False
        plane.refresh()
    pg.quit()

if __name__ == '__main__':
    main()
