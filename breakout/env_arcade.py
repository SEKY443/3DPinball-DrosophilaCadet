"""Breakout + real arcade power-ups: multiball and a laser (shooting bricks directly), the two
examples the user asked for instead of pulling in and adapting an unfamiliar third-party
open-source clone.

Why a fresh implementation instead of finding+modifying a GitHub Arkanoid clone: most such repos
couple physics tightly to a pygame render loop and keyboard polling, not a step(action) API, so
"adapt it" would mean rewriting most of it anyway; it would also mean running and vetting a
stranger's code (this project's own standing security practice - see the user's CLAUDE.md - is to
vet third-party code before running it, which a random small game repo usually gets zero
scrutiny). breakout/env.py already proved the pattern (pure Python/numpy, in-process, fast); this
extends it with two new mechanics rather than importing one.

Multiball mirrors env_python/pinball_env.py's real ball2 handling almost exactly: a second (and
third) ball can become active, and the circuit perceives it the same way - not raw coordinates,
but a summarized LC4/LPLC2 looming signal (vel2/size2), the same Giant-Fiber-inspired split as
the primary ball's vel/size. "Primary ball" for observation purposes is whichever active ball
slot is found FIRST when scanning (not a fixed slot index) - a deliberate fix for the exact bug
class pinball's engine had (see project history: BallList[0] being a fixed, potentially-stale
slot caused a real multiball double-feed bug there).

Laser: breaking a brick has a chance to grant a permanent (for the rest of the episode)
shoot capability - a third action bit fires a bullet upward from the paddle (rate-limited),
breaking whatever brick it hits.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from breakout.env import (
    BALL_RADIUS, BALL_SPEED, BRICK_GAP, BRICK_HEIGHT, BRICK_TOP_Y, FIELD_H, FIELD_W, LIVES,
    PADDLE_SPEED, PADDLE_Y, _size_signal, _velocity_signal,
)

PADDLE_HALF_WIDTH = 1.5
BRICK_ROWS = 4
BRICK_COLS = 8
BRICK_W = FIELD_W / BRICK_COLS

MAX_BALLS = 3
MULTIBALL_PROB = 0.15  # chance a broken brick spawns an extra ball, if a slot is free

MAX_BULLETS = 5
BULLET_SPEED = 0.6
BULLET_COOLDOWN_TICKS = 8
LASER_PROB = 0.10  # chance a broken brick grants the laser (once granted, stays for the episode)

ACT_LEFT, ACT_RIGHT, ACT_SHOOT = 0, 1, 2

# ball_x/y/vx/vy: the first currently-active ball slot found (see module docstring - NOT a fixed
# index). vel2/size2: looming signal for the second active ball found, if any (0 otherwise).
OBS_BALL_X, OBS_BALL_Y, OBS_BALL_VX, OBS_BALL_VY, OBS_PADDLE_X, OBS_HAS_LASER, OBS_VEL, OBS_SIZE, OBS_VEL2, OBS_SIZE2 = range(10)
OBS_DIM = 10
OBS_LABELS = ["ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_x", "has_laser", "vel", "size", "vel2", "size2"]


class ArcadeBreakoutEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.MultiBinary(3)
        self._rng = np.random.default_rng(seed)

        self.paddle_x = 0.0
        self.ball_x = np.zeros(MAX_BALLS)
        self.ball_y = np.zeros(MAX_BALLS)
        self.ball_vx = np.zeros(MAX_BALLS)
        self.ball_vy = np.zeros(MAX_BALLS)
        self.ball_active = np.zeros(MAX_BALLS, dtype=bool)

        self.bullet_x = np.zeros(MAX_BULLETS)
        self.bullet_y = np.zeros(MAX_BULLETS)
        self.bullet_active = np.zeros(MAX_BULLETS, dtype=bool)
        self._shoot_cooldown = 0

        self.has_laser = False
        self.bricks = np.ones((BRICK_ROWS, BRICK_COLS), dtype=bool)
        self.lives = LIVES
        self.score = 0

    def _spawn_ball(self, slot: int, x: float, y: float, angle: float) -> None:
        self.ball_x[slot] = x
        self.ball_y[slot] = y
        self.ball_vx[slot] = BALL_SPEED * np.sin(angle)
        self.ball_vy[slot] = BALL_SPEED * np.cos(angle)
        self.ball_active[slot] = True

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.paddle_x = FIELD_W / 2.0
        self.bricks[:, :] = True
        self.lives = LIVES
        self.score = 0
        self.has_laser = False
        self.bullet_active[:] = False
        self._shoot_cooldown = 0
        self.ball_active[:] = False
        self._spawn_ball(0, FIELD_W / 2.0, PADDLE_Y + 3.0, self._rng.uniform(-0.6, 0.6))
        return self._obs(), self._info(paddle_hit=False, brick_hit=False, ball_lost=False)

    def _active_ball_slots(self) -> np.ndarray:
        return np.flatnonzero(self.ball_active)

    def _obs(self) -> np.ndarray:
        active = self._active_ball_slots()
        if len(active) == 0:
            bx = by = bvx = bvy = 0.0
            vel = size = 0.0
            vel2 = size2 = 0.0
        else:
            primary = active[0]
            bx, by, bvx, bvy = self.ball_x[primary], self.ball_y[primary], self.ball_vx[primary], self.ball_vy[primary]
            vel = _velocity_signal(bvy)
            size = _size_signal(by)
            if len(active) > 1:
                secondary = active[1]
                vel2 = _velocity_signal(self.ball_vy[secondary])
                size2 = _size_signal(self.ball_y[secondary])
            else:
                vel2 = size2 = 0.0
        return np.array([bx, by, bvx, bvy, self.paddle_x, float(self.has_laser), vel, size, vel2, size2], dtype=np.float32)

    def _info(self, paddle_hit: bool, brick_hit: bool, ball_lost: bool, ball_brick_hit: bool = False) -> dict:
        return {
            "score": self.score,
            "paddle_hit": paddle_hit,
            "brick_hit": brick_hit,
            # Which brick-breaks were caused by the BALL (real paddle-interception skill) vs a
            # bullet (near-skill-free once the laser is held - just holding the shoot button
            # clears bricks directly above whatever column the paddle idles in). A first version
            # of this trainer credited both equally, and CEM found "ignore the ball, drain fast,
            # farm bullet kills" was cheaper than real paddle skill (confirmed: mean paddle_hits
            # collapsed to 0.6/episode with episodes ending in ~300-550 ticks instead of the
            # usual 1500-3000). Fitness now only credits this flag - see train_arcade.py.
            "ball_brick_hit": ball_brick_hit,
            "ball_lost": ball_lost,
            "lives": self.lives,
            "bricks_left": int(self.bricks.sum()),
            "balls_active": int(self.ball_active.sum()),
            "has_laser": self.has_laser,
        }

    def _break_brick(self, row: int, col: int) -> None:
        self.bricks[row, col] = False
        self.score += 10
        if not self.has_laser and self._rng.random() < LASER_PROB:
            self.has_laser = True
        if self._rng.random() < MULTIBALL_PROB:
            free = np.flatnonzero(~self.ball_active)
            if len(free):
                bx = col * BRICK_W + BRICK_W / 2.0
                by = BRICK_TOP_Y - row * (BRICK_HEIGHT + BRICK_GAP)
                self._spawn_ball(int(free[0]), bx, by, self._rng.uniform(-1.0, 1.0))

    def step(self, action: np.ndarray):
        left = bool(action[ACT_LEFT])
        right = bool(action[ACT_RIGHT])
        shoot = bool(action[ACT_SHOOT])
        if left and not right:
            self.paddle_x -= PADDLE_SPEED
        elif right and not left:
            self.paddle_x += PADDLE_SPEED
        self.paddle_x = float(np.clip(self.paddle_x, PADDLE_HALF_WIDTH, FIELD_W - PADDLE_HALF_WIDTH))

        if self._shoot_cooldown > 0:
            self._shoot_cooldown -= 1
        if shoot and self.has_laser and self._shoot_cooldown == 0:
            free = np.flatnonzero(~self.bullet_active)
            if len(free):
                slot = int(free[0])
                self.bullet_x[slot] = self.paddle_x
                self.bullet_y[slot] = PADDLE_Y + 0.3
                self.bullet_active[slot] = True
                self._shoot_cooldown = BULLET_COOLDOWN_TICKS

        bullet_brick_hit = self._step_bullets()
        paddle_hit = False
        ball_lost = False
        ball_brick_hit = False
        for slot in self._active_ball_slots():
            hit, lost = self._step_ball(slot)
            paddle_hit = paddle_hit or hit
            ball_lost = ball_lost or lost
            if self._check_brick_collision(slot):
                ball_brick_hit = True
        brick_hit = bullet_brick_hit or ball_brick_hit

        if ball_lost and not self.ball_active.any():
            self.lives -= 1
            if self.lives > 0:
                self._spawn_ball(0, FIELD_W / 2.0, PADDLE_Y + 3.0, self._rng.uniform(-0.6, 0.6))

        terminated = self.lives <= 0 or bool(self.bricks.sum() == 0)
        return self._obs(), 0.0, terminated, False, self._info(paddle_hit, brick_hit, ball_lost, ball_brick_hit)

    def _step_ball(self, slot: int) -> tuple[bool, bool]:
        self.ball_x[slot] += self.ball_vx[slot]
        self.ball_y[slot] += self.ball_vy[slot]

        if self.ball_x[slot] <= BALL_RADIUS:
            self.ball_x[slot] = BALL_RADIUS
            self.ball_vx[slot] = abs(self.ball_vx[slot])
        elif self.ball_x[slot] >= FIELD_W - BALL_RADIUS:
            self.ball_x[slot] = FIELD_W - BALL_RADIUS
            self.ball_vx[slot] = -abs(self.ball_vx[slot])
        if self.ball_y[slot] >= FIELD_H - BALL_RADIUS:
            self.ball_y[slot] = FIELD_H - BALL_RADIUS
            self.ball_vy[slot] = -abs(self.ball_vy[slot])

        paddle_hit = False
        ball_lost = False
        if self.ball_y[slot] <= PADDLE_Y + BALL_RADIUS and self.ball_vy[slot] < 0:
            if abs(self.ball_x[slot] - self.paddle_x) <= PADDLE_HALF_WIDTH + BALL_RADIUS:
                paddle_hit = True
                self.ball_y[slot] = PADDLE_Y + BALL_RADIUS
                offset = (self.ball_x[slot] - self.paddle_x) / PADDLE_HALF_WIDTH
                speed = float(np.hypot(self.ball_vx[slot], self.ball_vy[slot]))
                angle = np.clip(offset, -1.0, 1.0) * 0.7
                self.ball_vx[slot] = speed * np.sin(angle)
                self.ball_vy[slot] = speed * np.cos(angle)
            elif self.ball_y[slot] <= 0.0:
                ball_lost = True
                self.ball_active[slot] = False
        return paddle_hit, ball_lost

    def _check_brick_collision(self, slot: int) -> bool:
        if not self.ball_active[slot]:
            return False
        x, y = self.ball_x[slot], self.ball_y[slot]
        if y < BRICK_TOP_Y - BRICK_ROWS * (BRICK_HEIGHT + BRICK_GAP) or y > BRICK_TOP_Y + BRICK_HEIGHT:
            return False
        col = int(x // BRICK_W)
        row = int((BRICK_TOP_Y - y) // (BRICK_HEIGHT + BRICK_GAP))
        if not (0 <= row < BRICK_ROWS and 0 <= col < BRICK_COLS) or not self.bricks[row, col]:
            return False
        self._break_brick(row, col)
        self.ball_vy[slot] = -self.ball_vy[slot]
        return True

    def _step_bullets(self) -> bool:
        brick_hit = False
        for slot in np.flatnonzero(self.bullet_active):
            self.bullet_y[slot] += BULLET_SPEED
            if self.bullet_y[slot] > FIELD_H:
                self.bullet_active[slot] = False
                continue
            if self.bullet_y[slot] < BRICK_TOP_Y - BRICK_ROWS * (BRICK_HEIGHT + BRICK_GAP) or self.bullet_y[slot] > BRICK_TOP_Y + BRICK_HEIGHT:
                continue
            col = int(self.bullet_x[slot] // BRICK_W)
            row = int((BRICK_TOP_Y - self.bullet_y[slot]) // (BRICK_HEIGHT + BRICK_GAP))
            if 0 <= row < BRICK_ROWS and 0 <= col < BRICK_COLS and self.bricks[row, col]:
                self._break_brick(row, col)
                self.bullet_active[slot] = False
                brick_hit = True
        return brick_hit
