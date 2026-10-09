"""Parameter extraction from VNA data: the pipeline of Sec. IV B of the paper.

The simulation needs, per channel, the resonator parameters ``f0``, ``Qr``,
``Qc``, ``alpha`` and the rf-SQUID parameters ``lamb``, ``Mc``, ``Mfr`` and the
trapped flux. They come from a VNA flux sweep -- ``S21`` over a few MHz around
the resonance at each of about a hundred flux-bias points -- in these steps:

1. **Circle and notch fit at every flux point.** :func:`fit_resonance_trace`
   removes the cable delay (a line fitted to the phase in a band at each
   edge of the sweep, after dividing out the fitted resonance, iterated),
   then fits the notch model with the circle method of Probst et al. 2015
   (:mod:`~umux_cmt.fitting.circle`, :mod:`~umux_cmt.fitting.notch`),
   giving ``fr``, ``Qr``, ``|Qc|`` and ``alpha`` at that flux.
2. **Resonator parameters** are the medians of ``Qr``, ``|Qc|`` and ``alpha``
   over the flux points; ``Qc = |Qc| / cos(alpha)`` is the DCM coupling Q and
   ``1/Qi = 1/Qr - 1/Qc``.
3. **Flux-response fit.** ``fr`` against the bias current is fitted with the
   screened-SQUID model, Eq. (2) of the paper, by :func:`fit_flux_response`
   (differential evolution from several seeds, polished with least squares;
   :mod:`~umux_cmt.fitting.flux`): ``f0``, ``lamb``, ``Mc``, ``Mfr`` and the
   absolute flux offset ``I0``, which :func:`relative_I_trapped` turns into
   the ``trapped_current`` of the configuration files.
4. **Configuration.** :func:`summarize_channel` collects everything into the
   dictionary behind Table I, and :func:`umux_cmt.config.write_config` writes
   the TOML file the simulations load.

For a wide sweep of a whole chip, :func:`fit_all_resonances`
(:mod:`~umux_cmt.fitting.comb`) finds and fits every notch with one cable
delay, as for Fig. 4(a)-(b). Diagnostic plots are in
:mod:`umux_cmt.fitting.plots`, which is imported (with matplotlib) only when
it is first used.

Typical use::

    import numpy as np
    import umux_cmt as uc
    from umux_cmt import fitting

    sw = uc.load_vna_sweep("data/vna/res2_flux_power_sweep_2026-09-26.npz")
    k = sw.power_index(-26)                                  # extraction power
    fits = [fitting.fit_resonance_trace(sw.f, s) for s in sw.s21[k]]
    fr, Qr, Qc_abs, alpha = np.array([t.as_tuple() for t in fits]).T
    I = sw.bias_current(M_nominal=14.4e-12)
    ff = fitting.fit_flux_response(I, fr)
    table = fitting.summarize_channel(Qr, Qc_abs, alpha, ff)

The numerics are those of the code the paper was produced with (see the
tests in ``tests/test_fitting_equivalence.py``); only plotting and printing
were removed, and failures raise or return NaNs where the original did.
Where the pieces of the original code went (``resonator_fitting/include``
and the paper's parameter notebook ``res2_params``):

* ``circle_fitting.fit_circle`` -> :func:`fit_circle`
* ``resonance_fitting``: ``calculate_tau``, ``initial_fit``, ``fit_phase``,
  ``refined_fit``, ``residues`` -> the same names, in
  :mod:`~umux_cmt.fitting.notch`
* ``all_resonances``: ``find_all_peaks``, ``fit_all_resonators`` ->
  :func:`find_resonances`, :func:`fit_all_resonances`
* ``lambdafit``: ``phi_of_phie``, ``fr_of_I``, ``I_bare``, ``absolute_I0``,
  ``relative_I_trapped``, ``short_signal_periodicity``, ``guess_params``,
  ``fit_lambda`` -> the same names, in :mod:`~umux_cmt.fitting.flux`
* ``flux_response``: ``refined_flux_fit``, ``fit_all_flux_responses`` -> not
  ported (see :mod:`~umux_cmt.fitting.flux`)
* ``res2_params``: ``notch``, ``fit_trace``, ``fit_flux``, ``table_params`` ->
  :func:`notch_model`, :func:`fit_resonance_trace`, :func:`fit_flux_response`,
  :func:`summarize_channel`
"""
from .circle import fit_circle, circle_points
from .notch import (notch_model, phase_model, calculate_tau, initial_fit, fit_phase, refined_fit, residues,
                    NotchFitError, NotchFit, fit_resonance, NotchStages, fit_resonance_stages, TraceFit,
                    fit_resonance_trace)
from .comb import find_resonances, fit_all_resonances, CombFit
from .flux import (phi_of_phie, fr_of_I, I_bare, absolute_I0, relative_I_trapped, trapped_flux_phase,
                   short_signal_periodicity, guess_params, fit_lambda, FluxFit, fit_flux_response,
                   summarize_channel, LS_FIT, LS_DESIGN, Z1_DEFAULT)


def __getattr__(name):
    """Import :mod:`umux_cmt.fitting.plots` (and matplotlib) on first use rather than with the package."""
    if name == "plots":
        import importlib
        return importlib.import_module(".plots", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "fit_circle", "circle_points",
    "notch_model", "phase_model", "calculate_tau", "initial_fit", "fit_phase", "refined_fit", "residues",
    "NotchFitError", "NotchFit", "fit_resonance", "NotchStages", "fit_resonance_stages", "TraceFit",
    "fit_resonance_trace",
    "find_resonances", "fit_all_resonances", "CombFit",
    "phi_of_phie", "fr_of_I", "I_bare", "absolute_I0", "relative_I_trapped", "trapped_flux_phase",
    "short_signal_periodicity", "guess_params", "fit_lambda", "FluxFit", "fit_flux_response", "summarize_channel",
    "LS_FIT", "LS_DESIGN", "Z1_DEFAULT",
    "plots",
]
