"""Diagnostic plots for the fits in :mod:`umux_cmt.fitting`.

Every function takes the data and a fit result and draws on new or given
axes; nothing here is needed to compute anything.

::

    from umux_cmt import fitting
    from umux_cmt.fitting import plots

    tr = fitting.fit_resonance_trace(f, s21)
    plots.notch_panels(f, tr)                       # IQ circle, |S21| dB, phase

    ff = fitting.fit_flux_response(I, fr)
    plots.flux_response(ff)                         # fr(I) with the fit and residuals
"""
import matplotlib.pyplot as plt
import numpy as np

from .circle import circle_points
from .notch import NotchFit, TraceFit


def _window(f, fit):
    """Frequencies, data and model of the points that entered a notch fit."""
    if isinstance(fit, TraceFit):
        w = fit.window
        return f[w], fit.s21_corrected[w], fit.model(f[w])
    raise TypeError("expected a TraceFit; for a NotchFit pass the data with notch_panels(f, fit, s21=...)")


def notch_panels(f, fit, s21=None, axes=None, data_kw=None, fit_kw=None):
    """IQ circle, ``|S21|`` in dB and phase of one resonance with its fit.

    Parameters
    ----------
    f : ndarray
        Frequencies in Hz.
    fit : TraceFit or NotchFit
        The fit. For a :class:`~umux_cmt.fitting.notch.TraceFit` the
        delay-corrected trace it holds is drawn; for a
        :class:`~umux_cmt.fitting.notch.NotchFit` pass the raw data as ``s21``.
    s21 : ndarray of complex, optional
        Data, required with a ``NotchFit``.
    axes : sequence of 3 Axes, optional
        Axes to draw on; a new figure is made otherwise.

    Returns
    -------
    (fig, axes)
    """
    if isinstance(fit, NotchFit):
        if s21 is None:
            raise ValueError("pass the data as s21 with a NotchFit")
        ff, zz, mm = f, s21, fit.model(f)
    else:
        ff, zz, mm = _window(f, fit)
    data_kw = dict(ls="none", marker=".", ms=3, color="black") | (data_kw or {})
    fit_kw = dict(lw=1.2) | (fit_kw or {})
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    else:
        fig = axes[0].figure
    ax_iq, ax_mag, ax_ph = axes

    ax_iq.plot(zz.real, zz.imag, **data_kw)
    ax_iq.plot(mm.real, mm.imag, **fit_kw)
    ax_iq.set_xlabel("Re $S_{21}$")
    ax_iq.set_ylabel("Im $S_{21}$")
    ax_iq.set_aspect("equal", adjustable="datalim")

    x = (ff - fit.fr) / 1e3
    ax_mag.plot(x, 20 * np.log10(np.abs(zz)), **data_kw)
    ax_mag.plot(x, 20 * np.log10(np.abs(mm)), **fit_kw)
    ax_mag.set_xlabel("$f - f_r$ (kHz)")
    ax_mag.set_ylabel("$|S_{21}|$ (dB)")

    ax_ph.plot(x, np.unwrap(np.angle(zz)), **data_kw)
    ax_ph.plot(x, np.unwrap(np.angle(mm)), **fit_kw)
    ax_ph.set_xlabel("$f - f_r$ (kHz)")
    ax_ph.set_ylabel("arg $S_{21}$ (rad)")

    if fit.ok:
        ax_mag.set_title(f"$f_r$ = {fit.fr / 1e6:.3f} MHz, $Q_r$ = {fit.Qr:.0f}, $|Q_c|$ = {fit.Qc_abs:.0f}, "
                         f"$\\alpha$ = {fit.alpha:.3f}", fontsize="small")
    return fig, axes


def circle_fit(z, zc, r, ax=None, **kw):
    """The data ``z`` in the complex plane with the circle ``(zc, r)``.

    Parameters
    ----------
    z : ndarray of complex
        Data.
    zc : complex
        Centre of the circle.
    r : float
        Radius of the circle.
    ax : Axes, optional
        Axes to draw on; a new figure is made otherwise.
    **kw
        Passed to ``ax.plot`` for the circle.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot(z.real, z.imag, ls="none", marker=".", ms=3, color="black")
    xs, ys = circle_points(zc.real, zc.imag, r)
    ax.plot(xs, ys, **(dict(lw=1.2) | kw))
    ax.plot(zc.real, zc.imag, "+", color="gray")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    return ax


def circle_stages(stages, axes=None):
    """The three stages of the circle fit of one resonance.

    (a) the delay-corrected data with the fitted circle, (b) the data
    translated to the circle centre and rotated, with the refined circle,
    (c) the phase of the translated data against frequency with the
    arctangent fit that gives ``fr`` and ``Qr``.

    Parameters
    ----------
    stages : NotchStages
        From :func:`umux_cmt.fitting.fit_resonance_stages`.
    axes : sequence of 3 Axes, optional
        Axes to draw on; a new figure is made otherwise.

    Returns
    -------
    (fig, axes)
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    else:
        fig = axes[0].figure
    ax_a, ax_b, ax_c = axes
    circle_fit(stages.s21_corrected, stages.zc, stages.r, ax=ax_a)
    ax_a.set_title("delay-corrected $S_{21}$", fontsize="small")
    xc, yc, r = stages.circle_t
    circle_fit(stages.z_t, xc + 1j * yc, r, ax=ax_b)
    ax_b.set_title("translated and rotated", fontsize="small")
    x = (stages.f - stages.fit.fr) / 1e3
    ax_c.plot(x, np.angle(stages.z_t), ls="none", marker=".", ms=3, color="black")
    ax_c.plot(x, stages.phase_model(stages.f), lw=1.2)
    ax_c.set_xlabel("$f - f_r$ (kHz)")
    ax_c.set_ylabel("phase (rad)")
    ax_c.set_title(f"phase fit: $Q_r$ = {stages.phase_params[1]:.0f}", fontsize="small")
    return fig, axes


def flux_response(fit, axes=None, n_smooth=1000, every=1, label=None, color=None):
    """Measured ``fr(I)`` with the fitted model and the residuals.

    Parameters
    ----------
    fit : FluxFit
        From :func:`umux_cmt.fitting.fit_flux_response` (it holds the data).
    axes : (Axes, Axes), optional
        Axes for the response and the residuals; a new figure otherwise.
    n_smooth : int, optional
        Points of the smooth model curve.
    every : int, optional
        Plot every ``every``-th data point.
    label : str, optional
        Legend label of the model curve.
    color : optional
        Colour of the data, the model and the residuals.

    Returns
    -------
    (fig, axes)
    """
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    else:
        fig = axes[0].figure
    ax1, ax2 = axes
    I, fr = fit.I, fit.fr
    I_smooth = np.linspace(I[0], I[-1], n_smooth)
    ax1.plot(I[::every] * 1e3, fr[::every] / 1e6, "o", color=color, mfc="white", mew=0.8, ms=3.5)
    ax1.plot(I_smooth * 1e3, fit.model(I_smooth) / 1e6, color=color, lw=1.5, zorder=3, label=label)
    ax2.plot(I * 1e3, fit.residuals / 1e3, "o-", color=color, ms=2, lw=0.6)
    ax2.axhline(0, color="k", ls="--", lw=0.8)
    ax1.set_ylabel("$f_r$ (MHz)")
    ax2.set_ylabel("Res. (kHz)")
    ax2.set_xlabel("Line current (mA)")
    if label:
        ax1.legend()
    return fig, axes


def comb_overview(f, s21, comb, ax=None):
    """``|S21|`` of a wide sweep with the detected resonances marked.

    Parameters
    ----------
    f : ndarray
        Frequencies in Hz.
    s21 : ndarray of complex
        Transmission.
    comb : CombFit
        From :func:`umux_cmt.fitting.fit_all_resonances` on the same sweep.
    ax : Axes, optional
        Axes to draw on; a new figure is made otherwise.

    Returns
    -------
    Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 2.5))
    db = 20 * np.log10(np.abs(s21))
    ax.plot(f / 1e9, db, lw=0.6, color="black")
    ax.vlines(f[comb.peaks] / 1e9, db.min(), db.max(), color="tab:red", ls="--", lw=0.6, alpha=0.6)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("$|S_{21}|$ (dB)")
    return ax


def parameter_vs_flux(flux, Qr, Qc_abs, alpha, axes=None):
    """``Qr``, ``|Qc|`` and ``alpha`` of the notch fits against the flux bias, with their medians.

    Parameters
    ----------
    flux : ndarray
        Flux bias of each trace (any unit; the label says Phi0).
    Qr, Qc_abs, alpha : ndarray
        Notch-fit results at each flux point, e.g. from
        :func:`umux_cmt.fitting.fit_resonance_trace`.
    axes : sequence of 3 Axes, optional
        Axes to draw on; a new figure is made otherwise.

    Returns
    -------
    (fig, axes)
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
    else:
        fig = axes[0].figure
    for ax, y, name in zip(axes, (Qr, Qc_abs, alpha), ("$Q_r$", "$|Q_c|$", r"$\alpha$ (rad)")):
        ax.plot(flux, y, ".", ms=3, color="black")
        ax.axhline(np.nanmedian(y), color="tab:red", lw=1, label=f"median {np.nanmedian(y):.4g}")
        ax.set_xlabel("Flux bias ($\\Phi_0$)")
        ax.set_ylabel(name)
        ax.legend(fontsize="small")
    return fig, axes


__all__ = ["notch_panels", "circle_fit", "circle_stages", "flux_response", "comb_overview", "parameter_vs_flux"]
