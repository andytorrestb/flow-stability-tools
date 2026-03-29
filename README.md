# Flow Stability Tools

This repository contains a modular, configuration-driven framework for temporal linear stability analysis using the inviscid Rayleigh equation.

The current implementation runs a case pipeline with strict configuration validation and profile-wise alpha scans for:

- tanh_shear: U(z) = 0.5 * (1 + tanh(z))
- parabolic: U(z) = z^2
- bickley_jet: U(z) = sech(z)^2
- wake_deficit: U(z) = 1 - 0.6 * sech(z)^2
- asymmetric_mixing_layer: U(z) = 0.35 + 0.65 * tanh(z)
- double_shear_layer: U(z) = 0.5 * [tanh(2*(z+1)) - tanh(2*(z-1))]

## Current Functionality

- Strict YAML config validation with Pydantic.
- Case lifecycle with state progression:
	- INITIALIZED -> RUNNING -> COMPLETED
- Modular components for:
	- SymPy baseflow construction and symbolic derivatives
	- Chebyshev collocation operators on [-L, L]
	- Generalized eigenvalue Rayleigh solve
	- Alpha scanning and dominant-mode extraction
	- Plot generation for growth/frequency trends
- Case-isolated execution in cases/case_001.

## Physics and Outputs

The solver uses the temporal normal-mode convention:

q'(x,z,t) = q_hat(z) * exp(i * (alpha * x - omega * t))

with:

- growth_rate = Im(omega)
- frequency = Re(omega)

Rayleigh equation solved:

(U - c)(phi'' - alpha^2 phi) - U'' phi = 0, where c = omega / alpha

For each configured profile, the case computes:

- dominant growth curve max omega_i(alpha)
- dominant frequency curve omega_r(alpha)
- alpha_star
- omega_star
- growth_rate
- frequency
- symbolic baseflow expressions U(z), U'(z), U''(z)

## Project Layout

- src/config: schema and config loading
- src/core: case/context/state models
- src/pipeline: pipeline and step interfaces
- src/components: baseflow, spectral operators, Rayleigh solve, scanning, plotting
- src/steps: setup, solve, analysis
- src/runners: case runner orchestration
- cases/case_001: runnable example case
- cases/case_003..case_006: profile-specific validation cases for newly added profiles

## Run the Example Case

1. Install dependencies:

python -m pip install -r requirements.txt

2. Run the case:

python cases/case_001/run.py

## Run Regression Checks Across All Cases

To execute every case folder under `cases/` and validate core artifacts:

python run_all_cases_regression.py

To run only specific cases:

python run_all_cases_regression.py --case case_001 --case case_003

## Case Configuration

Primary config file: cases/case_001/config.yaml

Config supports:

- solver:
	- type: rayleigh_inviscid
	- profiles: [tanh_shear, parabolic, bickley_jet, wake_deficit, asymmetric_mixing_layer, double_shear_layer]
- numerical:
	- L
	- N
	- eigenvalue_mag_threshold
- alpha_scan:
	- alpha_min
	- alpha_max
	- alpha_count
- output:
	- enable_plots
	- growth_plot_filename
	- frequency_plot_filename
	- show_symbolic_in_console
	- include_symbolic_in_profile_summaries
	- include_symbolic_latex
- refinement:
	- enabled
	- profile
	- N_base
	- multiplier
- pipeline steps ordering

## Expected Artifacts

After a successful run, the case writes artifacts to cases/case_001/results:

- run_metadata.json
- scan_results.json
- profile_summaries.json
- growth_vs_alpha.png (if plotting enabled)
- frequency_vs_alpha.png (if plotting enabled)

When symbolic reporting is enabled, profile summaries include a `sympy` section with:

- symbol
- U, U_prime, U_double_prime
- U_pretty, U_prime_pretty, U_double_prime_pretty
- latex forms when `include_symbolic_latex` is true

## Notes

- tanh_shear may show unstable alpha bands depending on N/L and scan window.
- parabolic has no inflection point, so strong positive inviscid growth should be treated cautiously and checked with refinement/sensitivity.
