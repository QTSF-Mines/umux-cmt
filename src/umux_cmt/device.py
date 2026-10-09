"""A multiplexer: resonator-SQUID channels hanging off one feedline."""
import numpy as np

from .channel import Channel


class Multiplexer:
    """A set of :class:`~umux_cmt.channel.Channel` objects on a common feedline.

    The device is the object a simulation is run on. It hands the solver the
    per-channel loss rates and detuning functions
    (:meth:`compile_static_parameters`) and the asymmetry angles
    (:meth:`get_alpha_vec`). Build one by hand with :meth:`add`, or from a TOML
    configuration file with :func:`umux_cmt.config.load_config`.

    Parameters
    ----------
    name : str, optional
        A label for printouts.
    feedline_Z0 : float, optional
        Characteristic impedance of the feedline in Ohm.
    """

    def __init__(self, name="uMux", feedline_Z0=50.0):
        self.name = name
        self.Z0 = feedline_Z0
        self.channels = []

    def __repr__(self):
        return f"Multiplexer({self.name!r}, {len(self.channels)} channel(s))"

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, i):
        return self.channels[i]

    def add(self, channel):
        """Append a channel to the feedline."""
        if not isinstance(channel, Channel):
            raise TypeError("only Channel objects can be added to a Multiplexer")
        self.channels.append(channel)
        return channel

    def compile_static_parameters(self, omega_p, flux_ramp_current_func=lambda t: 0, input_current_func=lambda t: 0):
        """Per-channel inputs for the solver.

        Parameters
        ----------
        omega_p : float
            Probe tone angular frequency in rad/s.
        flux_ramp_current_func, input_current_func : callable, optional
            Currents ``I(t)`` in A on the ``'FluxRamp'`` and ``'Input'`` lines
            of every SQUID.

        Returns
        -------
        (delta_funcs, kappa_i, kappa_c)
            An object array of ``delta(t, Er)`` callables and the arrays of
            internal and coupling loss rates in rad/s, one entry per channel.
        """
        channels = self.channels
        kappa_i_vec = np.array([ch.get_kappa_i() for ch in channels])
        kappa_c_vec = np.array([ch.get_kappa_c() for ch in channels])
        delta_func_vec = np.empty(len(channels), dtype=object)
        for i, ch in enumerate(channels):
            delta_func_vec[i] = ch.get_Delta_func(omega_p, flux_ramp_current_func, input_current_func)
        return delta_func_vec, kappa_i_vec, kappa_c_vec

    def get_alpha_vec(self):
        """Per-channel coupling asymmetry angles in rad, ordered like the channels."""
        return np.array([ch.alpha for ch in self.channels])

    def get_f0_vec(self):
        """Per-channel bare resonant frequencies in Hz."""
        return np.array([ch.f0 for ch in self.channels])

    def summary(self):
        """A short table of the channels, as a string."""
        lines = [f"{self.name}: {len(self.channels)} channel(s), Z0 = {self.Z0:g} Ohm",
                 f"{'ch':>3} {'f0 (MHz)':>10} {'Qr':>7} {'Qc':>7} {'Qi':>9} {'alpha':>7} {'lamb':>6} {'Mc (pH)':>8} {'BW (kHz)':>9}"]
        for i, ch in enumerate(self.channels):
            lines.append(f"{i:3d} {ch.f0 / 1e6:10.3f} {ch.Qr:7.0f} {ch.Qc:7.0f} {ch.Qi:9.0f} {ch.alpha:7.3f} "
                         f"{ch.squid.lamb:6.3f} {ch.Mc * 1e12:8.2f} {ch.linewidth / 1e3:9.0f}")
        return "\n".join(lines)
