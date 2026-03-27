# Flow Stability Tools

This repository contains a modular, configuration-driven framework for temporal linear stability analysis using the inviscid Rayleigh equation.

The current implementation runs a case pipeline with strict configuration validation and profile-wise alpha scans for:

- tanh_shear: U(z) = 0.5 * (1 + tanh(z))
- parabolic: U(z) = z^2

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

## Project Layout

- src/config: schema and config loading
- src/core: case/context/state models
- src/pipeline: pipeline and step interfaces
- src/components: baseflow, spectral operators, Rayleigh solve, scanning, plotting
- src/steps: setup, solve, analysis
- src/runners: case runner orchestration
- cases/case_001: runnable example case

## Run the Example Case

1. Install dependencies:

python -m pip install -r requirements.txt

2. Run the case:

python cases/case_001/run.py

## Case Configuration

Primary config file: cases/case_001/config.yaml

Config supports:

- solver:
	- type: rayleigh_inviscid
	- profiles: [tanh_shear, parabolic]
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

## Notes

- tanh_shear may show unstable alpha bands depending on N/L and scan window.
- parabolic has no inflection point, so strong positive inviscid growth should be treated cautiously and checked with refinement/sensitivity.
