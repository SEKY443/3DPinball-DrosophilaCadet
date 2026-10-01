"""A small, self-contained Breakout physics simulation as a Gymnasium env - the practice-project
counterpart to env_python/pinball_env.py.

Deliberately pure Python/numpy, no native engine or IPC socket: Breakout's physics (a ball, a
paddle, a brick grid) is simple enough to simulate directly in-process, unlike pinball's real
SpaceCadetPinball engine. This means training workers just hold a BreakoutEnv object each - no
subprocess spawning, no IPC crash/orphan-process class of bugs pinball's pipeline needed a whole
ResilientPool to work around.

Observation channels (see build_connectome.py's OBS_LABELS): ball_x, ball_y, ball_vx, ball_vy,
paddle_x (all raw kinematics), vel and size (a looming signal for the ball closing in on the
paddle's y-plane, the same LC4/LPLC2 Giant-Fiber-inspired split used for pinball's ball2 channel -
see env_python/pinball_env.py's _velocity_signal/_size_signal, ported here almost unchanged since
paddle interception is an even more direct looming-response analog than flipper timing was).

Action space: MultiBinary(2) - [move_left, move_right], independent bits (both pressed cancels
out), matching PinballEnv's MultiBinary convention that fixed_circuit_agent.py's MultiBinaryDecoder
already handles.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

FIELD_W = 20.0
FIELD_H = 24.0

PADDLE_Y = 1.0
PADDLE_HALF_WIDTH = 1.5
PADDLE_SPEED = 0.6

BALL_RADIUS = 0.3
BALL_SPEED = 0.35

BRICK_ROWS = 4
BRICK_COLS = 8
BRICK_TOP_Y = 22.0
BRICK_HEIGHT = 0.8
BRICK_GAP = 0.1
BRICK_W = FIELD_W / BRICK_COLS

LIVES = 3

ACT_LEFT, ACT_RIGHT = 0, 1

OBS_BALL_X, OBS_BALL_Y, OBS_BALL_VX, OBS_BALL_VY, OBS_PADDLE_X, OBS_VEL, OBS_SIZE = range(7)
OBS_DIM = 7

# Same shape as PinballEnv's ball2 loom channels (env_python/pinball_env.py) - LC4-style velocity
# signal (capped, only counts closing speed) and LPLC2-style size signal (Gaussian, peaks near
# the paddle's y-plane). LOOM_ZONE_Y/SIZE_PEAK_DISTANCE/SIZE_SIGMA are in the same field-height
# units as the rest of this env, not pinball's table-coordinate units.
VEL_CAP = 20.0
LOOM_ZONE_Y = PADDLE_Y + 4.0
SIZE_PEAK_DISTANCE = 2.0
SIZE_SIGMA = 2.5


def _velocity_signal(ball_vy: float) -> float:
    # Only closing speed (moving DOWN toward the paddle) counts as looming - a ball moving away
    # (positive vy, after a bounce) isn't an approaching threat. Scaled up from BALL_SPEED's tiny
    # per-tick units onto the same [0, VEL_CAP] range pinball's loom channels use, so the two
    # projects' OBS_SCALE conventions stay comparable.
    closing = max(-ball_vy, 0.0)
    return float(min(closing / BALL_SPEED * 5.0, VEL_CAP))


def _size_signal(ball_y: float) -> float:
    remaining = ball_y - PADDLE_Y
    if remaining > LOOM_ZONE_Y - PADDLE_Y:
        return 0.0
    return float(np.exp(-((remaining - SIZE_PEAK_DISTANCE) ** 2) / (2 * SIZE_SIGMA ** 2)))


class BreakoutEnv(gym.Env):
    """Difficulty knobs (paddle_half_width, brick_rows/cols, brick_fill_prob) exist so a trained
    champion can be stress-tested against a harder/differently-shaped field it never saw during
    training, without retraining - see breakout/harder_demo.py. Note the observation space never
    includes brick positions at all (see OBS_LABELS) - the circuit only ever tracks the ball, so
    varying the brick layout tests paddle-size/ball-trajectory robustness specifically, not
    whether the policy "knows about" bricks."""

    metadata = {"render_modes": []}

    def __init__(self, seed: Optional[int] = None, paddle_half_width: float = PADDLE_HALF_WIDTH,
                 brick_rows: int = BRICK_ROWS, brick_cols: int = BRICK_COLS, brick_fill_prob: float = 1.0):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.MultiBinary(2)
        self._rng = np.random.default_rng(seed)
        self.paddle_half_width = paddle_half_width
        self.brick_rows = brick_rows
        self.brick_cols = brick_cols
        self.brick_fill_prob = brick_fill_prob
        self.brick_w = FIELD_W / brick_cols
        self.paddle_x = 0.0
        self.ball_x = self.ball_y = self.ball_vx = self.ball_vy = 0.0
        self.bricks = np.ones((brick_rows, brick_cols), dtype=bool)
        self.lives = LIVES
        self.score = 0

    def _reset_ball(self) -> None:
        self.ball_x = FIELD_W / 2.0
        self.ball_y = PADDLE_Y + 3.0
        angle = self._rng.uniform(-0.6, 0.6)
        self.ball_vx = BALL_SPEED * np.sin(angle)
        self.ball_vy = BALL_SPEED * np.cos(angle)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.paddle_x = FIELD_W / 2.0
        if self.brick_fill_prob >= 1.0:
            self.bricks[:, :] = True
        else:
            self.bricks = self._rng.random((self.brick_rows, self.brick_cols)) < self.brick_fill_prob
        self.lives = LIVES
        self.score = 0
        self._reset_ball()
        return self._obs(), self._info(paddle_hit=False, brick_hit=False, ball_lost=False)

    def _obs(self) -> np.ndarray:
        return np.array([
            self.ball_x, self.ball_y, self.ball_vx, self.ball_vy, self.paddle_x,
            _velocity_signal(self.ball_vy), _size_signal(self.ball_y),
        ], dtype=np.float32)

    def _info(self, paddle_hit: bool, brick_hit: bool, ball_lost: bool) -> dict:
        return {
            "score": self.score,
            "paddle_hit": paddle_hit,
            "brick_hit": brick_hit,
            "ball_lost": ball_lost,
            "lives": self.lives,
            "bricks_left": int(self.bricks.sum()),
        }

    def step(self, action: np.ndarray):
        left = bool(action[ACT_LEFT])
        right = bool(action[ACT_RIGHT])
        if left and not right:
            self.paddle_x -= PADDLE_SPEED
        elif right and not left:
            self.paddle_x += PADDLE_SPEED
        self.paddle_x = float(np.clip(self.paddle_x, self.paddle_half_width, FIELD_W - self.paddle_half_width))

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_x <= BALL_RADIUS:
            self.ball_x = BALL_RADIUS
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x >= FIELD_W - BALL_RADIUS:
            self.ball_x = FIELD_W - BALL_RADIUS
            self.ball_vx = -abs(self.ball_vx)
        if self.ball_y >= FIELD_H - BALL_RADIUS:
            self.ball_y = FIELD_H - BALL_RADIUS
            self.ball_vy = -abs(self.ball_vy)

        brick_hit = self._check_brick_collision()

        paddle_hit = False
        ball_lost = False
        if self.ball_y <= PADDLE_Y + BALL_RADIUS and self.ball_vy < 0:
            if abs(self.ball_x - self.paddle_x) <= self.paddle_half_width + BALL_RADIUS:
                paddle_hit = True
                self.ball_y = PADDLE_Y + BALL_RADIUS
                self.ball_vy = abs(self.ball_vy)
                # Hit offset steers the return angle, same "aim with where you hit it" skill
                # real breakout rewards - without this the paddle is a pure mirror and hitting
                # a specific brick is never a controllable choice.
                offset = (self.ball_x - self.paddle_x) / self.paddle_half_width
                speed = float(np.hypot(self.ball_vx, self.ball_vy))
                angle = np.clip(offset, -1.0, 1.0) * 0.7
                self.ball_vx = speed * np.sin(angle)
                self.ball_vy = speed * np.cos(angle)
            elif self.ball_y <= 0.0:
                ball_lost = True
                self.lives -= 1
                if self.lives > 0:
                    self._reset_ball()

        terminated = self.lives <= 0 or bool(self.bricks.sum() == 0)
        return self._obs(), 0.0, terminated, False, self._info(paddle_hit, brick_hit, ball_lost)

    def _check_brick_collision(self) -> bool:
        if self.ball_y < BRICK_TOP_Y - self.brick_rows * (BRICK_HEIGHT + BRICK_GAP) or self.ball_y > BRICK_TOP_Y + BRICK_HEIGHT:
            return False
        col = int(self.ball_x // self.brick_w)
        row = int((BRICK_TOP_Y - self.ball_y) // (BRICK_HEIGHT + BRICK_GAP))
        if not (0 <= row < self.brick_rows and 0 <= col < self.brick_cols):
            return False
        if not self.bricks[row, col]:
            return False
        self.bricks[row, col] = False
        self.score += 10
        self.ball_vy = -self.ball_vy
        return True
