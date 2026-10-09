# Known issues and findings

Things found while preparing this release that are worth knowing before you build on the code. None of
them was changed in the package, because the numbers of the paper were produced with the code as it is
and the tests pin that behaviour; each is documented at the place in the code where it applies.

## The tabulated screening coefficients deviate from the exact expansion at order λ⁴

`umux_cmt/squid.py` carries the 36 coefficients of the series of Wegner, Enss and Kempf (2022), Table 1,
as they were used for the paper. In the zero-RF-flux limit the exact screening response follows from
Kepler's equation as a Kapteyn series,

    S(φ_ext) = Σ_c 2 (−1)^(c+1) J_c(c λ) cos(c φ_ext),

whose Taylor coefficients agree with the table only through the λ³ terms: 32 of the 36 tabulated values
differ from the exact expansion from λ⁴ upward (for example the λ⁵ cos φ_ext coefficient is 1/16 in the
table and 1/192 exactly). The resulting error is smooth and small for the devices of the paper: 0.07% of
the modulation swing at λ = 0.2, 0.27% at λ = 0.32 (about 2 kHz of resonance shift for the paper's channel,
under 1% of its linewidth, and part of the −6.7 kHz offset discussed in Sec. V A), and 1.4% at λ = 0.5.
`umux_cmt.screening_exact` gives the exact low-power reference; the test
`test_screening_series_against_exact_reference` bounds the deviation at 0.3% of the swing for λ ≤ 0.32.
Whether the deviation is in the published table or in its transcription has not been established.

## Absolute tolerance of the steady-state sweep

`steady_state_sweep` runs with `rtol = 1e-5, atol = 1e-9` (in units of the field, √J), the values of the
paper. Below about −90 dBm at the chip the field on resonance is itself only a few 1e-10, so the absolute
tolerance is then not small compared with the solution; the fitted resonance of a sweep moves by about
0.5 kHz (0.1% of the linewidth) when the tolerance is tightened. `atol_rel` scales the tolerance with the
field, as `simulate` does by default.

## Phase wrapping in the first stage of the notch fit

`fitting.fit_phase` fits `np.angle` of the rotated circle without unwrapping, as the original code did.
For asymmetry angles beyond about 0.2 rad the phase can cross ±π inside the fit window, and the full fit
then starts from a wrong minimum (large χ², `Qr` far off). The channel of the paper (α ≈ −0.09 rad) is not
affected; a few channels of the whole-chip survey in example `02` are, and are flagged there.

## Edge-band delay fit: convergence stop versus a fixed eight iterations

`fit_resonance_trace` is the parameter notebook's `fit_trace`: the delay line is refitted until it changes
by less than 5 ps, at most eight times. The paper's power-dependence notebook used a variant without the
convergence test (always eight iterations). The difference is below 0.05 Hz in `fr` and 1e-6 dB in the
fitted attenuation; pass `tol=0` to reproduce the fixed-iteration variant exactly.

## Circle fit on noise-free data (fixed in this release)

The algebraic circle fit selects the smallest positive generalised eigenvalue. For points that lie exactly
on a circle (simulated, noise-free `S21`) that eigenvalue is zero up to round-off and could come out slightly
negative, in which case the original code silently returned a wrong circle. The port accepts eigenvalues
down to −1e-9 of the largest; this changes nothing for measured data, where the eigenvalue is positive
(the equivalence tests confirm it).
