"""Model constants, copied exactly from Shiu et al. `model.py` (`default_params`)."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LIFParams:
    v_rest: float = -52.0        # mV   resting potential            (v_0)
    v_reset: float = -52.0       # mV   reset potential after spike  (v_rst)
    v_thresh: float = -45.0      # mV   spike threshold              (v_th)
    tau_m: float = 20.0          # ms   membrane time constant       (t_mbr)
    tau_syn: float = 5.0         # ms   synapse time constant        (tau)
    t_refrac: float = 2.2        # ms   refractory period            (t_rfc)
    t_delay: float = 1.8         # ms   synaptic delay               (t_dly)
    w_syn: float = 0.275         # mV   weight per synapse           (w_syn)
    dt: float = 0.1              # ms   time step (Brian2 default, used by Shiu)
    poisson_gain: float = 250.0  #      Poisson kick = w_syn * 250   (f_poi)

    def as_dict(self) -> dict:
        return asdict(self)


SHIU = LIFParams()
