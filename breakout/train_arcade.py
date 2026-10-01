#!/usr/bin/env python3
"""CEM neuroevolution for the fixed-circuit agent on ArcadeBreakoutEnv (breakout/env_arcade.py) -
the multiball + laser power-up variant. Same method as breakout/train.py, separate script since
the connectome/observation shape differs (10 channels vs breakout.env's 7) and can't warm-start
from the plain-Breakout checkpoint.

Usage:
    python breakout/train_arcade.py --workers 4 --population 24 --generations 500
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "nfly"))

from agents.fixed_circuit_agent import FixedCircuitAgent  # noqa: E402

from breakout.env_arcade import ArcadeBreakoutEnv  # noqa: E402

# ball_x, ball_y, ball_vx, ball_vy, paddle_x, has_laser, vel, size, vel2, size2 - see
# breakout/env_arcade.py's OBS_LABELS/OBS_DIM.
OBS_SCALE = (20.0, 24.0, 1.0, 1.0, 20.0, 1.0, 20.0, 1.0, 20.0, 1.0)

PADDLE_HIT_BONUS = 10.0
BALL_LOST_PENALTY = 3.0

# Fitness isolates to PADDLE_HIT alone now - the same flipper-focus lesson from
# agents/train_pinball_circuit_cem.py, rediscovered here. A first version credited BRICK_BONUS
# for ball_brick_hit (a brick broken by the ball, as opposed to a bullet) on top of
# PADDLE_HIT_BONUS - looked fixed (removed the bullet-farming exploit) but a real eval still
# found mean paddle_hits stuck at 0.8/episode, with one episode clearing all 32 bricks on ZERO
# paddle hits. Root cause: reset() spawns the ball already heading UP toward the bricks (the
# angle range is centered on straight up), so it can hit bricks purely via wall bounces - and
# multiball multiplies how many independently-bouncing balls can do this - without the paddle
# ever touching it. Any credit tied to "a brick broke" is exploitable this way regardless of
# which entity (bullet or ball) is nominally credited, because the ball's own passive physics
# can trigger it too. Isolating fitness to PADDLE_HIT specifically (an event that provably
# requires the paddle to be under the ball) removes the loophole at its root instead of patching
# each new way to reach the same intermediate signal.

_WORKER_ENV = None
_WORKER_AGENT = None
_WORKER_MAX_STEPS = 3000


def _worker_init(connectome_path: str, readout_dim: int, max_steps: int, seed_base: int) -> None:
    global _WORKER_ENV, _WORKER_AGENT, _WORKER_MAX_STEPS
    torch.set_num_threads(1)
    _WORKER_MAX_STEPS = max_steps
    _WORKER_AGENT = FixedCircuitAgent.from_json(connectome_path, readout_dim=readout_dim, obs_scale=OBS_SCALE)
    _WORKER_ENV = ArcadeBreakoutEnv(seed=seed_base + os.getpid())
    _WORKER_AGENT.calibrate(_WORKER_ENV.observation_space)


def _evaluate(flat_weights: np.ndarray) -> float:
    from agents.train_pinball_circuit_cem import set_flat_params

    agent, env = _WORKER_AGENT, _WORKER_ENV
    set_flat_params(agent.decoder, flat_weights)
    obs, info = env.reset()
    h = agent.initial_state(1)
    fitness = 0.0
    steps = 0
    done = False
    with torch.no_grad():
        while not done and steps < _WORKER_MAX_STEPS:
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            action, h = agent.act(obs_t, h, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            if info["paddle_hit"]:
                fitness += PADDLE_HIT_BONUS
            if info["ball_lost"]:
                fitness -= BALL_LOST_PENALTY
            steps += 1
            done = terminated or truncated
    return fitness


def gaussian_weights(rng: np.random.Generator, n: int, scale: float = 0.5) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32) * scale


def get_flat_params(decoder: torch.nn.Module) -> np.ndarray:
    return torch.cat([p.detach().flatten() for p in decoder.parameters()]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connectome", default=os.path.join(os.path.dirname(__file__), "arcade", "connectome.json"))
    parser.add_argument("--readout-dim", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--elites", type=int, default=6)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--courses-per-candidate", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "checkpoints", "breakout_circuit_cem_arcade.pt"))
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--fresh-fraction", type=float, default=0.2)
    parser.add_argument("--sigma-floor", type=float, default=0.15)
    parser.add_argument("--validation-episodes", type=int, default=4)
    args = parser.parse_args()

    ref_agent = FixedCircuitAgent.from_json(args.connectome, readout_dim=args.readout_dim, obs_scale=OBS_SCALE)
    n_params = get_flat_params(ref_agent.decoder).size
    print(f"decoder parameter count (CEM search dimension): {n_params}", flush=True)

    rng = np.random.default_rng(args.seed)
    sigma = np.full(n_params, 0.8, dtype=np.float32)
    champion = gaussian_weights(rng, n_params)
    mean = np.zeros(n_params, dtype=np.float32)
    champion_fitness = -1e18
    start_gen = 1

    if args.resume:
        ckpt = torch.load(args.resume, weights_only=False)
        mean, sigma = ckpt["mean"], ckpt["sigma"]
        champion, champion_fitness = ckpt["champion"], ckpt["champion_fitness"]
        start_gen = ckpt["generation"] + 1
        print(f"resumed from {args.resume} at generation {start_gen} (champion_fitness={champion_fitness:.2f})", flush=True)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=args.workers, initializer=_worker_init,
                     initargs=(args.connectome, args.readout_dim, args.max_steps, args.seed))

    try:
        for generation in range(start_gen, args.generations + 1):
            t0 = time.time()
            n_fresh = round(args.fresh_fraction * args.population)
            n_perturbed = args.population - 1 - n_fresh
            population = (
                [champion.copy()]
                + [mean + sigma * gaussian_weights(rng, n_params, scale=1.0) for _ in range(n_perturbed)]
                + [gaussian_weights(rng, n_params, scale=0.7) for _ in range(n_fresh)]
            )
            tasks = [w for w in population for _ in range(args.courses_per_candidate)]
            results = pool.map(_evaluate, tasks)
            fitness = np.array(results, dtype=np.float32).reshape(args.population, args.courses_per_candidate).mean(axis=1)

            order = np.argsort(-fitness)
            elite_idx = order[:args.elites]
            elite_weights = np.stack([population[i] for i in elite_idx])
            mu = args.elites
            log_ranks = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            rank_weights = (log_ranks / log_ranks.sum()).astype(np.float32)
            new_mean = (rank_weights[:, None] * elite_weights).sum(axis=0)
            new_sigma = np.sqrt((rank_weights[:, None] * (elite_weights - new_mean) ** 2).sum(axis=0))
            mean = 0.3 * mean + 0.7 * new_mean
            sigma = np.maximum(args.sigma_floor, 0.3 * sigma + 0.7 * new_sigma)

            best_idx = order[0]
            if fitness[best_idx] > champion_fitness:
                validation = float(np.mean(pool.map(_evaluate, [population[best_idx]] * args.validation_episodes)))
                if validation > champion_fitness:
                    champion_fitness = validation
                    champion = population[best_idx].copy()

            elapsed = time.time() - t0
            print(
                f"gen {generation:4d}  best {fitness[best_idx]:8.2f}  mean {fitness.mean():8.2f}  "
                f"champion {champion_fitness:8.2f}  sigma_mean {sigma.mean():.4f}  {elapsed:5.1f}s",
                flush=True,
            )

            if generation % args.save_every == 0 or generation == args.generations:
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                torch.save({
                    "mean": mean, "sigma": sigma, "champion": champion, "champion_fitness": champion_fitness,
                    "generation": generation, "readout_dim": args.readout_dim,
                }, args.out)
                print(f"  saved checkpoint: {args.out}", flush=True)
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    print(f"training complete, final checkpoint: {args.out}", flush=True)


if __name__ == "__main__":
    main()
