"""Run the synthetic Standard Gamble benchmark and regenerate simulation_results.tex."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np


U = np.linspace(0.02, 0.98, 49)
GAMMA = np.linspace(0.45, 2.15, 35)
RHO = np.linspace(0.55, 1.45, 10)
OFFERS = np.linspace(0.02, 0.98, 49)
REWARDS = np.array([
    (low, low + fraction * (high - low), high, p)
    for low, high, fraction, p in product(
        (0.0, 0.25), (0.7, 1.0), (0.25, 0.50, 0.75), (0.2, 0.4, 0.6, 0.8)
    )
])
TOTAL = 36
CALIBRATION = 12
WIDTH = 0.20
LAPSE = 0.04
SLOPE = 16
METHODS = ("Deterministic SG", "Prob. bisection", "Bayes (flat)", "Two-stage HA-SG")
SCENARIOS = (
    ("Matched", 0.63, 1.20, LAPSE, 1.0),
    ("High lapses", 0.63, 1.20, 0.16, 1.0),
    ("Population shift", 0.36, 0.85, LAPSE, 1.0),
    ("Task shift", 0.63, 1.20, LAPSE, 1.35),
)


def normalise(weights, axis=None):
    return weights / weights.sum(axis=axis, keepdims=True)


def response_probability(difference, lapse):
    return lapse + (1 - 2 * lapse) / (1 + np.exp(-SLOPE * difference))


def reward_response(offers, gamma, rho, lapse):
    low, certain, high, p = np.moveaxis(offers, -1, 0)
    lower = low**rho
    difference = lower + p**gamma * (high**rho - lower) - certain**rho
    return response_probability(difference, lapse)


STATE_YES = response_probability(
    OFFERS[:, None, None] ** GAMMA[None, None, :] - U[None, :, None], LAPSE
)
REWARD_YES = reward_response(
    REWARDS[:, None, None, :], GAMMA[None, :, None], RHO[None, None, :], LAPSE
)
REWARD_YES_FLAT = REWARD_YES.reshape(len(REWARDS), -1)
PBA_YES = np.where(
    OFFERS[:, None] > U[None, :], 0.88,
    np.where(OFFERS[:, None] < U[None, :], 0.12, 0.5),
)


def entropy(probability):
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    return -probability * np.log(probability) - (1 - probability) * np.log1p(-probability)


def information_offer(posterior, chance_yes):
    """Choose the offer giving most information about the first grid dimension."""
    marginal = posterior.sum(axis=1)
    yes_by_target = np.einsum("ij,kij->ki", posterior, chance_yes)
    mean_yes = yes_by_target.sum(axis=1)
    conditional_yes = yes_by_target / marginal
    return int(np.argmax(entropy(mean_yes) - (entropy(conditional_yes) * marginal).sum(axis=1)))


def update(posterior, chance_yes, answer):
    return normalise(posterior * (chance_yes if answer else 1 - chance_yes))


def interval(grid, mass):
    cumulative = np.cumsum(mass)
    lower, upper = np.searchsorted(cumulative, [0.025, 0.975])
    return grid[lower], grid[upper]


def subject(rng, mean_u, mean_gamma):
    return (
        np.clip(rng.normal(mean_u, 0.12), 0.06, 0.94),
        np.clip(rng.normal(mean_gamma, 0.23), 0.50, 1.55),
        rng.uniform(0.65, 1.35),
    )


def gaussian_kernels(grid, means, scale):
    return normalise(np.exp(-0.5 * ((grid[None, :] - means[:, None]) / scale)**2), axis=1)


def fit_population(rng, history_size):
    """Fit a two-level grid model with Gibbs sampling of group mean parameters."""
    u_kernels = gaussian_kernels(U, np.linspace(0.24, 0.78, 13), 0.12)
    gamma_kernels = gaussian_kernels(GAMMA, np.linspace(0.65, 1.60, 13), 0.23)
    log_mass = np.zeros((len(u_kernels), len(gamma_kernels)))
    for _ in range(history_size):
        u, gamma, rho = subject(rng, 0.63, 1.20)
        state_offers = rng.integers(len(OFFERS), size=24)
        reward_offers = rng.integers(len(REWARDS), size=24)
        state_answers = rng.random(24) < response_probability(OFFERS[state_offers]**gamma - u, LAPSE)
        reward_answers = rng.random(24) < reward_response(REWARDS[reward_offers], gamma, rho, LAPSE)
        state_likelihood = np.prod(
            np.where(state_answers[:, None, None], STATE_YES[state_offers], 1 - STATE_YES[state_offers]),
            axis=0,
        )
        reward_likelihood = np.prod(
            np.where(reward_answers[:, None, None], REWARD_YES[reward_offers], 1 - REWARD_YES[reward_offers]),
            axis=0,
        ).mean(axis=1)
        log_mass += np.log((u_kernels @ state_likelihood) @ (gamma_kernels * reward_likelihood).T)

    mass = normalise(np.exp(log_mass - log_mass.max()))
    u_cdf = normalise(mass, axis=0).cumsum(axis=0)
    gamma_cdf = normalise(mass, axis=1).cumsum(axis=1)
    counts = np.zeros_like(mass)
    for _ in range(4):
        gamma_index = rng.integers(mass.shape[1])
        for step, (r_u, r_gamma) in enumerate(rng.random((2200, 2))):
            u_index = np.searchsorted(u_cdf[:, gamma_index], r_u)
            gamma_index = np.searchsorted(gamma_cdf[u_index], r_gamma)
            if step >= 200:
                counts[u_index, gamma_index] += 1

    predictive = u_kernels.T @ normalise(counts) @ gamma_kernels
    exact_predictive = u_kernels.T @ mass @ gamma_kernels
    discrepancy = 0.5 * np.abs(predictive - exact_predictive).sum()
    return normalise(predictive), float(discrepancy)


def calibrate(prior, gamma, rho, lapse, uniforms):
    posterior = np.broadcast_to(prior[:, None] / len(RHO), (len(GAMMA), len(RHO)))
    for uniform in uniforms:
        index = information_offer(posterior, REWARD_YES)
        answer = uniform < reward_response(REWARDS[index], gamma, rho, lapse)
        posterior = update(posterior, REWARD_YES[index], answer)
    return posterior.sum(axis=1)


def evaluate_subject(population, u, gamma_r, rho, lapse, gamma_s, state_uniforms, reward_uniforms):
    flat = np.full((len(U), len(GAMMA)), 1 / (len(U) * len(GAMMA)))
    gamma_prior = population.sum(axis=0)
    calibrated = calibrate(gamma_prior, gamma_r, rho, lapse, reward_uniforms)
    calibrated_state = normalise((population / gamma_prior) * calibrated)
    threshold = u ** (1 / gamma_s)
    calibration_error = abs(float(calibrated @ threshold**GAMMA) - u)

    outcomes = []
    for method in range(len(METHODS)):
        offset = CALIBRATION if method == 3 else 0
        if method == 0:
            lower, upper = 0.0, 1.0
        elif method == 1:
            threshold_mass = np.full(len(U), 1 / len(U))
        else:
            posterior = calibrated_state if method == 3 else flat
        first_precise = None

        for step, uniform in enumerate(state_uniforms[:TOTAL - offset], offset + 1):
            if method == 0:
                offer = (lower + upper) / 2
            elif method == 1:
                index = int(np.searchsorted(threshold_mass.cumsum(), 0.5))
                offer = OFFERS[index]
            else:
                index = information_offer(posterior, STATE_YES)
                offer = OFFERS[index]
            answer = uniform < response_probability(offer**gamma_s - u, lapse)

            if method == 0:
                if answer:
                    upper = offer
                else:
                    lower = offer
                estimate = (lower + upper) / 2
            elif method == 1:
                threshold_mass = update(threshold_mass, PBA_YES[index], answer)
                lower, upper = interval(U, threshold_mass)
                estimate = U @ threshold_mass
            else:
                posterior = update(posterior, STATE_YES[index], answer)
                marginal = posterior.sum(axis=1)
                lower, upper = interval(U, marginal)
                estimate = U @ marginal
            if first_precise is None and upper - lower <= WIDTH + 1e-12:
                first_precise = step

        outcomes.append((
            estimate - u,
            lower <= u <= upper,
            first_precise if first_precise is not None else TOTAL,
            first_precise is not None,
        ))
    return outcomes, calibration_error


def run(seed, cohorts, cohort_size, history_size):
    rng = np.random.default_rng(seed)
    results = {scenario[0]: [] for scenario in SCENARIOS}
    calibration = {scenario[0]: [] for scenario in SCENARIOS}
    discrepancies = []
    for _ in range(cohorts):
        population, discrepancy = fit_population(rng, history_size)
        discrepancies.append(discrepancy)
        for name, mean_u, mean_gamma, lapse, task_shift in SCENARIOS:
            for _ in range(cohort_size):
                u, gamma_r, rho = subject(rng, mean_u, mean_gamma)
                outcomes, error = evaluate_subject(
                    population, u, gamma_r, rho, lapse, gamma_r * task_shift,
                    rng.random(TOTAL), rng.random(CALIBRATION),
                )
                results[name].append(outcomes)
                calibration[name].append(error)
    return results, calibration, discrepancies


def write_tables(output, results, calibration, discrepancies, seed, cohorts, cohort_size, history_size):
    lines = [
        "% Generated by simulate.py.",
        rf"\newcommand{{\SimSeed}}{{{seed}}}",
        rf"\newcommand{{\SimN}}{{{cohorts * cohort_size}}}",
        rf"\newcommand{{\SimCohorts}}{{{cohorts}}}",
        rf"\newcommand{{\SimHistory}}{{{history_size}}}",
        rf"\newcommand{{\SimBudget}}{{{TOTAL}}}",
        rf"\newcommand{{\SimCalibration}}{{{CALIBRATION}}}",
        rf"\newcommand{{\SimWidth}}{{{WIDTH:.2f}}}",
        rf"\newcommand{{\SimDiscrepancy}}{{{max(discrepancies):.4f}}}",
        "\\newcommand{\\SimResultRows}{%",
    ]
    for scenario, outcomes in results.items():
        for i, (method, values) in enumerate(zip(METHODS, np.asarray(outcomes).transpose(1, 0, 2))):
            error, covered, trials, reached = values.T
            lines.append(
                f"{scenario if i == 0 else ''} & {method} & {np.sqrt(np.mean(error**2)):.3f}"
                f" & {error.mean():+.3f} & {100 * covered.mean():.1f}"
                f" & {np.median(trials):.0f} & {100 * reached.mean():.1f} \\\\"
            )
        if scenario != next(reversed(results)):
            lines.append("\\midrule")
    lines.append("}")
    lines.append("\\newcommand{\\SimCalibrationRows}{%")
    lines.extend(f"{name} & {np.mean(errors):.3f} \\\\" for name, errors in calibration.items())
    lines.append("}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    cohorts, cohort_size, history_size = 4, 80, 36
    results, calibration, discrepancies = run(args.seed, cohorts, cohort_size, history_size)
    output = Path(__file__).with_name("simulation_results.tex")
    write_tables(output, results, calibration, discrepancies, args.seed, cohorts, cohort_size, history_size)
    print(f"Wrote {output.name} with {cohorts * cohort_size} synthetic subjects per scenario.")
