# Initial Velocity Profiles and Academic References

This note maps candidate initial/base velocity profiles to representative academic references and the physical situations they idealize.

## 1) Hyperbolic-Tangent Mixing Layer
- Profile: `U(z) = tanh(z/delta)` (or shifted/scaled equivalent)
- Physical situation: Two parallel streams with different speeds separated by a shear interface.
- Reference:
  - Michalke, A. (1964). On the inviscid instability of the hyperbolic-tangent velocity profile. *Journal of Fluid Mechanics*, 19(4), 543-556.

## 2) Bickley Jet (sech^2 jet)
- Profile: `U(z) = U0 * sech^2(z/delta)`
- Physical situation: Idealized free shear jet into quiescent surroundings.
- References:
  - Bickley, W. G. (1937). The plane jet. *Philosophical Magazine*, 23(156), 727-731.
  - Huerre, P., & Rossi, M. (1998). Hydrodynamic instabilities in open flows. In *Hydrodynamics and Nonlinear Instabilities* (Les Houches).

## 3) Gaussian Jet
- Profile: `U(z) = U0 * exp(-(z/delta)^2)`
- Physical situation: Rounded-core jet profile commonly used in model studies and fits.
- Reference:
  - Huerre, P., & Monkewitz, P. A. (1990). Local and global instabilities in spatially developing flows. *Annual Review of Fluid Mechanics*, 22, 473-537.

## 4) Wake (Velocity Deficit)
- Profile: `U(z) = U_inf - DeltaU * sech^2(z/delta)` (or Gaussian deficit variants)
- Physical situation: Bluff-body wake with centerline velocity deficit.
- Reference:
  - Monkewitz, P. A. (1988). The absolute and convective nature of instability in two-dimensional wakes at low Reynolds numbers. *Physics of Fluids*, 31(5), 999-1006.

## 5) Double Shear Layer
- Profile: Difference of two shifted tanh layers (scaled as needed)
- Physical situation: Two nearby shear interfaces, e.g., finite-thickness jet edge pair.
- Reference:
  - Hooper, A. P., & Huerre, P. (1984). Perturbed free shear layers. *Annual Review of Fluid Mechanics*, 16, 365-424.

## 6) Piecewise-Linear Shear (Smoothed for Numerics)
- Profile: Idealized linear segments with smooth transitions for spectral methods.
- Physical situation: Engineered or stratified shear with approximately constant-vorticity regions.
- Reference:
  - Drazin, P. G., & Reid, W. H. (2004). *Hydrodynamic Stability* (2nd ed.). Cambridge University Press.

## 7) Vortex-Sheet / Sharp-Shear Limit (Regularized)
- Profile: Very sharp tanh layer as a smooth approximation to a discontinuity.
- Physical situation: Near-vortex-sheet Kelvin-Helmholtz limit.
- References:
  - Chandrasekhar, S. (1961). *Hydrodynamic and Hydromagnetic Stability*. Oxford University Press.
  - Drazin, P. G., & Reid, W. H. (2004). *Hydrodynamic Stability* (2nd ed.). Cambridge University Press.

## 8) Parabolic/Channel-Like Profile
- Profile: `U(z) = 1 - (z/h)^2` (or equivalent scaling)
- Physical situation: Internal shear flow with centerline maximum (Poiseuille-like archetype).
- Reference:
  - Orszag, S. A. (1971). Accurate solution of the Orr-Sommerfeld stability equation. *Journal of Fluid Mechanics*, 50(4), 689-703.

## 9) Inflectional Polynomial S-Profile
- Profile: `U(z) = z - beta*z^3` (bounded/smoothed over finite domain)
- Physical situation: Controlled inflection-strength model for threshold and mode-competition studies.
- Reference:
  - Schmid, P. J., & Henningson, D. S. (2001). *Stability and Transition in Shear Flows*. Springer.

## 10) Asymmetric Mixing Layer
- Profile: `U(z) = U1 + (U2-U1)/2 * [1 + tanh((z-zc)/delta)]`, with unequal stream speeds
- Physical situation: Realistic open shear layers with speed asymmetry.
- References:
  - Dimotakis, P. E. (1991). Turbulent free shear layer mixing and combustion. In *High Speed Flight Propulsion Systems*.
  - Huerre, P., & Monkewitz, P. A. (1990). Local and global instabilities in spatially developing flows. *Annual Review of Fluid Mechanics*, 22, 473-537.

## Notes
- These references are intended as canonical starting points for profile motivation and linear-instability context.
- For implementation details in this repository, map each profile to symbolic `U(z)`, `U'(z)`, and `U''(z)` definitions in the baseflow component.
