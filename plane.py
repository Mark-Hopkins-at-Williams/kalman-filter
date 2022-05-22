import numpy as np
import os
import pygame as pg


def load_image(filename, colorkey=-1, scale=1):
    image = pg.image.load(filename)
    image = image.convert()
    size = image.get_size()
    size = (size[0] * scale, size[1] * scale)
    image = pg.transform.scale(image, size)
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, pg.RLEACCEL)
    return image, image.get_rect()


class Slider:

    def __init__(self, x, y, width, label, initial_percentage=0.5):
        self.x, self.y, self.width = x, y, width
        self.slider_x = self.x + initial_percentage * self.width - 2
        self.grace = 2
        self.label = label
        self.dragging = False

    def current_percentage(self):
        return (self.slider_x - self.x) / (self.width - 4)

    def draw(self, screen):
        pg.draw.rect(screen, "gray", (self.x, self.y, self.width, 10))
        pg.draw.rect(screen, "red", (self.slider_x, self.y-5, 4, 20))
        myfont = pg.font.SysFont("monospace", 15)
        text = myfont.render(self.label, 1, (255, 255, 255))
        screen.blit(text, (self.x, self.y+20))

    def notify(self, event):
        if event.type == pg.MOUSEBUTTONDOWN and not self.dragging:
            event_x, event_y = event.pos
            in_x_range = self.slider_x - self.grace <= event_x <= self.slider_x + 4 + self.grace
            in_y_range = self.y - 5 - self.grace <= event_y <= self.y + 15 + self.grace
            if in_x_range and in_y_range:
                self.dragging = True
        elif event.type == pg.MOUSEBUTTONUP:
            self.dragging = False
        elif self.dragging:
            try:
                event_x, _ = event.pos
                self.slider_x = max(self.x, event_x)
                self.slider_x = min(self.x + self.width - 4, self.slider_x)
            except Exception:
                pass


class CartesianPlane:
    def __init__(self, x_max, y_max, screen_width, screen_height,
                 bg_color=(0, 30, 105),
                 grid_color=(0, 0, 200)):
        self.screen = pg.display.set_mode((screen_width, screen_height), pg.SCALED)
        self.background = pg.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(bg_color)
        self.screen_width, self.screen_height = self.screen.get_size()
        self.x_max = x_max
        self.y_max = y_max
        self.x_pixel_increment = self.screen_width // self.x_max
        self.y_pixel_increment = self.screen_height // self.y_max
        for y in range(self.y_pixel_increment, self.screen_height, self.y_pixel_increment):
            pg.draw.aaline(self.background, grid_color, (0, y), (self.screen_width, y))
        for x in range(self.x_pixel_increment, self.screen_width, self.x_pixel_increment):
            pg.draw.aaline(self.background, grid_color, (x, 0), (x, self.screen_height))
        self.screen.blit(self.background, (0, 0))
        pg.display.flip()
        self.sprite_list = []
        self.sprites = pg.sprite.RenderPlain(self.sprite_list)
        self.screen = pg.display.get_surface()
        self.widgets = []

    def add_sprite(self, sprite):
        self.sprite_list.append(sprite)
        self.sprites = pg.sprite.RenderPlain(self.sprite_list)

    def add_widget(self, widget):
        self.widgets.append(widget)

    def refresh(self):
        self.screen.blit(self.background, (0, 0))
        bg = pg.image.load("images/oceanfloor.png")
        self.screen.blit(bg, (0, self.screen_height - 190))
        self.sprites.update()
        for sprite in self.sprites:
            sprite.redraw()
            x, y = sprite.current_position()
            coords = self.translate_coordinates(x, y)
            if coords is not None:
                width, height = sprite.size()
                sprite.rect = coords[0] - width//2, coords[1] - height//2
        self.sprites.draw(self.screen)
        for widget in self.widgets:
            widget.draw(self.screen)
        pg.display.flip()

    def notify(self, event):
        for sprite in self.sprites:
            sprite.notify(event)
        for widget in self.widgets:
            widget.notify(event)

    def in_bounds(self, x, y):
        return 0 <= x <= self.x_max, 0 <= y <= self.y_max

    def translate_coordinates(self, x, y):
        return (x * self.x_pixel_increment,
                self.screen_height - (y * self.y_pixel_increment))


class AnimatedSprite(pg.sprite.Sprite):

    def __init__(self, initial_xy, animation_cells, cell_scale):
        pg.sprite.Sprite.__init__(self)
        self.x, self.y = initial_xy
        self.animation_cells = animation_cells
        self.current_cell = 0
        self.cell_scale = cell_scale
        self.image, self.rect = load_image(self.animation_cells[self.current_cell], scale=self.cell_scale)
        self.move_divisor = 10
        self.move_queue = []

    def size(self):
        return self.rect.width, self.rect.height

    def current_position(self):
        return self.x, self.y

    def move(self, delta_x, delta_y):
        step_size = (delta_x / self.move_divisor, delta_y / self.move_divisor)
        self.move_queue += [step_size] * self.move_divisor

    def is_stationary(self):
        return len(self.move_queue) == 0

    def notify(self, event):
        pass

    def redraw(self):
        self.image, self.rect = load_image(self.animation_cells[self.current_cell], scale=self.cell_scale)

    def update(self):
        if len(self.move_queue) > 0:
            (delta_x, delta_y), self.move_queue = self.move_queue[0], self.move_queue[1:]
            self.current_cell = (self.current_cell + 1) % len(self.animation_cells)
            self.x, self.y = self.x + delta_x, self.y + delta_y

