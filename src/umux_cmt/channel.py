"""One uMux readout channel: the lumped-element resonator model of Sec. II A.

A channel is a quarter-wave resonator, capacitively coupled to the feedline,
whose shorted end is terminated by an inductor ``Lc`` screened by an rf-SQUID.
The CMT equation of motion needs three things from it:

* the loss rates ``kappa_i = omega_0/Qi`` and ``kappa_c = omega_0/Qc``,
* the asymmetry angle ``alpha`` of the complex coupling,
* the instantaneous resonant frequency ``omega_r(t, Er)``, which carries all of
  the SQUID non-linearity.

The resonant frequency comes from the effective LC model, Eq. (7) of the paper::

    omega_r = 1 / sqrt(Ceff * (Lres + Leff)),      Ceff = 4 / (pi*omega_0*Z0),

calibrated so that ``omega_r = omega_0`` when the SQUID does not load the
resonator. ``Leff`` depends on the external flux and on the RF flux amplitude,
which is computed from the energy stored in the cavity, Eq. (9).
"""
import numpy as np

from .squid import rfSQUID


class Channel:
    """One resonator-SQUID channel hanging off the feedline.

    Parameters
    ----------
    f0 : float
        Bare resonant frequency in Hz, i.e. the resonance when the SQUID does
        not load the resonator (``cos(phi_tot) = 0``). Fixed; the lumped model
        is calibrated against it. The instantaneous resonance is
        :meth:`get_omega_r`.
    Qi, Qc : float
        Internal and coupling quality factors. ``Qc`` is the real-part
        (diameter-correction-method) coupling Q, ``|Qc_hat| / cos(alpha)``,
        which sets the rate at which energy leaks into the feedline.
    Lc : float
        Terminating inductance in H.
    Mc : float
        Mutual inductance between the terminating inductor and the SQUID in H.
    squid : rfSQUID
        The SQUID screening ``Lc``.
    alpha : float, optional
        Asymmetry angle of the complex coupling in rad (Khalil et al. 2012).
    Z0 : float, optional
        Characteristic impedance of the resonator and feedline in Ohm.
    name : str, optional
        A label, used in printouts.
    Cc : float, optional
        Coupling capacitance in F. Not used by the model; kept as metadata.
    """

    def __init__(self, f0, Qi, Qc, Lc, Mc, squid, alpha=0.0, Z0=50.0, name="Channel", Cc=None):
        if not isinstance(squid, rfSQUID):
            raise TypeError("squid must be an rfSQUID")
        self.name = name
        self.f0 = f0
        self.Qi = Qi
        self.Qc = Qc
        self.Qr = 1 / (1 / Qi + 1 / Qc)
        self.alpha = alpha
        self.Lc = Lc
        self.Mc = Mc
        self.Cc = Cc
        self.Z0 = Z0
        self.squid = squid
        self.Ceff = None
        self.Lres = None
        self._calibrate()

    def __repr__(self):
        return (f"Channel({self.name!r}, f0={self.f0 / 1e6:.3f} MHz, Qr={self.Qr:.0f}, "
                f"Qc={self.Qc:.0f}, Qi={self.Qi:.0f}, alpha={self.alpha:.3f})")

    # ------------------------------------------------------------------ access

    @property
    def squids(self):
        """``[self.squid]``; kept for code written against the original framework."""
        return [self.squid]

    @property
    def flux_lines(self):
        """The SQUID's flux lines, by name."""
        return self.squid.flux_lines

    def line(self, name):
        """The flux line called ``name`` (``'FluxRamp'``, ``'Input'``, ``'TrappedFlux'``)."""
        return self.squid.flux_lines[name]

    # ------------------------------------------------------------------ rates

    @property
    def omega_0(self):
        """Bare angular resonant frequency ``2*pi*f0`` in rad/s."""
        return 2 * np.pi * self.f0

    def get_kappa_i(self):
        """Internal loss rate ``omega_0/Qi`` in rad/s."""
        return 2 * np.pi * self.f0 / self.Qi

    def get_kappa_c(self):
        """Coupling rate ``omega_0/Qc`` in rad/s (real part of the complex coupling)."""
        return 2 * np.pi * self.f0 / self.Qc

    @property
    def kappa(self):
        """Total loss rate ``kappa_i + kappa_c`` in rad/s."""
        return self.get_kappa_i() + self.get_kappa_c()

    @property
    def linewidth(self):
        """Resonator bandwidth ``f0/Qr`` in Hz (``kappa/2pi``)."""
        return self.kappa / (2 * np.pi)

    @property
    def ring_up_time(self):
        """Ring-up time ``2/kappa = 2*Qr/omega_r`` in s, Eq. (6) of the paper."""
        return 2 / self.kappa

    # ------------------------------------------------------------ inductance

    def get_Leff(self, Phi_rf=0.0):
        """Effective termination inductance ``Leff = Lc - (Mc**2/Ls) * S`` in H.

        Parameters
        ----------
        Phi_rf : float or ndarray, optional
            RF flux amplitude at the SQUID in Wb. Defaults to the low-power
            limit.
        """
        return self.Lc - (self.Mc ** 2 / self.squid.Ls) * self.squid.get_screening_factor(Phi_rf)

    def _calibrate(self):
        """Calibrate the lumped equivalent ``(Ceff, Lres)`` against ``f0``.

        ``f0`` is the resonance at the SQUID's calibration state: every flux
        LINE at zero, ``squid.Phi_ext`` left alone. The configuration loader sets
        ``Phi_ext`` so that this state has ``cos(phi_tot) = 0``, i.e. the SQUID
        not loading the resonator. ``Lres`` is whatever the line must carry to
        land there, computed through the same :meth:`get_Leff` the equation of
        motion uses, so the two agree exactly at that flux.

        Called once at construction. Changing ``squid.Phi_ext`` afterwards would
        silently desynchronise the model from ``f0``; call this again if you do.
        """
        squid = self.squid
        saved = {name: line.I for name, line in squid.flux_lines.items()}
        for line in squid.flux_lines.values():
            line.I = 0.0

        Leff_0 = self.get_Leff()
        omega_0 = 2 * np.pi * self.f0
        self.Ceff = 4.0 / (np.pi * omega_0 * self.Z0)
        self.Lres = 1.0 / (omega_0 ** 2 * self.Ceff) - Leff_0

        for name, I in saved.items():
            squid.flux_lines[name].I = I

    # ------------------------------------------------------------- resonance

    def get_Phi_rf(self, Er):
        """RF flux amplitude at the SQUID for a stored cavity energy ``Er`` (J), Eq. (9).

        The stored energy sets the RF current in the termination,
        ``I_T = sqrt(8*omega*Er/(pi*Z0))``, and hence ``Phi_rf = Mc*I_T``. The
        bare ``f0`` is used in that conversion rather than the instantaneous
        resonance (Sec. II A); for a 5 GHz resonator moving by 1 MHz that is an
        error of 0.02%.
        """
        return self.Mc * np.sqrt(16 * self.f0 * Er / self.Z0)

    def get_omega_r(self, Er):
        """Instantaneous angular resonant frequency at stored energy ``Er``.

        Because ``Er`` is the instantaneous energy ``|a|**2``, a detuned drive
        sees a smaller RF flux on its own, which is what produces the
        asymmetric high-power lineshapes.

        Parameters
        ----------
        Er : float or ndarray
            Stored photon field energy ``|a|**2`` in J.

        Returns
        -------
        float or ndarray
            ``omega_r`` in rad/s.
        """
        Phi_rf = self.Mc * np.sqrt(16 * self.f0 * Er / self.Z0)
        return 1.0 / np.sqrt((self.Lres + self.get_Leff(Phi_rf)) * self.Ceff)

    def get_fr(self, Er=0.0):
        """Instantaneous resonant frequency ``omega_r/2pi`` in Hz."""
        return self.get_omega_r(Er) / (2 * np.pi)

    def get_Delta_func(self, omega_p, flux_ramp_current_func=None, input_current_func=None):
        """Build the detuning ``Delta(t, Er) = omega_p - omega_r(t, Er)`` the solver calls.

        Parameters
        ----------
        omega_p : float
            Probe tone angular frequency in rad/s.
        flux_ramp_current_func : callable, optional
            ``I_fr(t)`` in A, applied to the ``'FluxRamp'`` line.
        input_current_func : callable, optional
            ``I_in(t)`` in A, applied to the ``'Input'`` line.

        Returns
        -------
        callable
            ``delta(t, Er)`` in rad/s. Evaluating it sets the line currents of
            the SQUID to their values at ``t``.
        """
        squid = self.squid
        fr_line = squid.flux_lines.get('FluxRamp')
        in_line = squid.flux_lines.get('Input')

        def delta_t(t, Er):
            if flux_ramp_current_func is not None and fr_line is not None:
                fr_line.I = flux_ramp_current_func(t)
            if input_current_func is not None and in_line is not None:
                in_line.I = input_current_func(t)
            return omega_p - self.get_omega_r(Er)

        return delta_t
