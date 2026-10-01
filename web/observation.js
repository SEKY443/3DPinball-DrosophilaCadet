// JS port of env_python/pinball_env.py::_state_to_obs - must match exactly (order, constants,
// gating) since the trained agent was calibrated on that exact 15-value observation. Verified
// against real-engine test vectors by web/test/verify_policy.mjs.

export const OBS_DIM = 15;

const BIN_CENTERS = [-2.5, 0.0, 2.5];
const BIN_SIGMA = 2.0;
// Ground-truth flipper-zone threshold (TFlipperEdge::YMin/YMax) - see pinball_env.py's
// LOOM_ZONE_Y comment.
const LOOM_ZONE_Y = 9.0;
const SIZE_PEAK_DISTANCE = 1.5;
const SIZE_SIGMA = 1.5;
const VEL_ATTENTION_SIGMA = 6.0;
const VEL_CAP = 20.0;

function spatialWeight(ballX, center) {
	return Math.exp(-((ballX - center) ** 2) / (2 * BIN_SIGMA ** 2));
}

// LC4's channel: retinal angular velocity, gated by distance to the flipper zone - see
// pinball_env.py's _velocity_signal docstring for why raw closing speed alone is wrong.
function velocitySignal(ballVy, ballY, active = true) {
	if (!active) return 0.0;
	const closing = Math.min(Math.max(ballVy, 0.0), VEL_CAP);
	const distance = LOOM_ZONE_Y - ballY;
	if (distance < 0) return 0.0;
	const attention = Math.exp(-(distance ** 2) / (2 * VEL_ATTENTION_SIGMA ** 2));
	return closing * attention;
}

// LPLC2's channel: Gaussian-tuned retinal size, peaked at SIZE_PEAK_DISTANCE before contact.
function sizeSignal(ballY, active = true) {
	if (!active) return 0.0;
	const distance = LOOM_ZONE_Y - ballY;
	if (distance < 0) return 0.0;
	return Math.exp(-((distance - SIZE_PEAK_DISTANCE) ** 2) / (2 * SIZE_SIGMA ** 2));
}

// state: { ball_x, ball_y, ball_vx, ball_vy, flipper_left, flipper_right, tilted,
//          ball2_y, ball2_vy, ball2_active } - same field names as env_python's State.
// Returns the 15-value observation array in OBS_BALL_X..OBS_TILTED order (pinball_env.py).
export function stateToObs(state) {
	const vel1 = velocitySignal(state.ball_vy, state.ball_y);
	const size1 = sizeSignal(state.ball_y);
	const binWeights = BIN_CENTERS.map((c) => spatialWeight(state.ball_x, c));
	return [
		state.ball_x, state.ball_y, state.ball_vx, state.ball_vy,
		binWeights[0] * vel1, binWeights[1] * vel1, binWeights[2] * vel1,
		binWeights[0] * size1, binWeights[1] * size1, binWeights[2] * size1,
		velocitySignal(state.ball2_vy, state.ball2_y, state.ball2_active),
		sizeSignal(state.ball2_y, state.ball2_active),
		state.flipper_left ? 1.0 : 0.0,
		state.flipper_right ? 1.0 : 0.0,
		state.tilted ? 1.0 : 0.0,
	];
}
