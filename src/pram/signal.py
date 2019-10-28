import matplotlib.pyplot as plt
import numpy as np

from attr                          import attrs, attrib
from pandas.plotting               import autocorrelation_plot
from statsmodels.tsa.stattools     import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ----------------------------------------------------------------------------------------------------------------------
# @attrs(slots=True)
# class DiscSpec(object):
#     '''
#     Discretization specs for a single series.
#     '''
#
#     bins  : list  = attrib(factory=list, converter=converters.default_if_none(factory=list))
#     right : float = attrib(default=False, converter=bool)
#     names : list  = attrib(factory=list, converter=converters.default_if_none(factory=list))


# ----------------------------------------------------------------------------------------------------------------------
class Signal(object):
    '''
    Time series considered a signal.

    When considered over time, mass dynamics and other time series that PramPy outputs can be seen as signals.  Signal
    processing toolkit can therefore be used to process those signals.  Because PramPy currently is a discrete-time
    system (i.e., it only supports iteration over time steps), all signals it outputs are also discrete.  However, with
    a sufficiently small time step size those signals can be a reasonably good approximation of the continuous-time
    data-generating processes.  One example of that are systems of ordinary differential equations (ODEs).  Because
    ODEs need to be numerically integrated (PramPy does not support analytical solvers), the time step size typically
    needs to be very small.

    A signal is a list of numpy arrays refered to as series.  What every single of those array denotes depends on the
    outside algorithm that creates the signal.  For example, they might (and often will) contain the time series of
    mass locus of a simulation.
    '''

    def __init__(self, series=None, names=None):
        if series is not None and not isinstance(signal, np.ndarray):
            raise ValueError('S needs to be an instance of ndarray.')

        self.series = series
        self.names = names
        # self.iter_max = 0

        if self.series is not None:
            if self.names is None:
                self.names = [None] * len(self.series)

            if len(self.series) != len(self.names):
                raise ValueError('The number of series and names must be identical.')

            # self.iter_max = max([len(s) for s in self.series])

    def add_series(self, s, name=None):
        if self.series is None or len(self.series) == 0:
            self.series = s
            self.names = [name]
        else:
            self.series = np.append(self.series, s, axis=0)
            self.names.append(name)

    # S,S,S,I,I,R,R,R,R,R  (3,2,5)

    # def discretize(self, disc_specs, right=False):
    #     '''
    #     If 'disc_spec' is a DiscSpec object, those discretization specs are used for all series.
    #     Otherwise, the size of the 'disc_spec' iterable is expected to be equal the number of series.
    #     '''
    #
    #     if isinstance(disc_specs, DiscSpec):
    #         disc_specs = [disc_specs] * len(self.series)
    #     elif len(disc_spec) == 1 and isinstance(disc_specs[0], DiscSpec):
    #         disc_specs = [disc_specs[0]] * len(self.series)
    #     elif len(disc_spec) != len(self.series):
    #         raise ValueError(f'The number of discretization specs must be either one or be equal to the number of signal series (i.e., {len(self.series)} for the current signal).')
    #
    #     return [np.digitize(self.series[i], disc_specs[i].bins, disc_specs[i].right) for i in len(self.series)]

    # def discretize_ns(self, bins):
    #     return self.discretize(self.get_bins(n, min, max))

    # def make_bins(self, n, min=0.0, max=1.0):
    #     '''
    #     Returns 'n' equally-sized bins on the interval defined. The defaults for 'min' and 'max' reflect the
    #     probabilistic nature of mass dynamics in PRAM.  These values can be changed because the Signal class can be
    #     used with arbitary time series.
    #     '''
    #
    #     if n <= 1:
    #         raise ValueError('The number of states needs to be at least two.')
    #     return self.discretize([min + ((max - min) / float(n)) * i for i in range(n)] + [max])

    def plot_autocorr(self, size, filename=None, do_ret_plot=False, dpi=100):
        fig, axes = plt.subplots(len(self.series), 1, figsize=size, sharex=True, dpi=dpi)
        fig.subplots_adjust(hspace=0, wspace=0)
        # plt.suptitle('', fontweight='bold')
        # plt.xlabel('Time', fontweight='bold')

        for i in range(len(self.series)):
            ax = axes[i]
            autocorrelation_plot(self.series[i,1:], ax=ax)
            ax.set_ylabel(self.names[i])

        if filename is None:
            plt.show()
        else:
            fig.savefig(filepath, dpi=dpi)

        return fig if do_ret_plot else self

        # fig, axes = plt.subplots(3,1,figsize=(16,3), dpi=100)
        # autocorrelation_plot(s[0], ax=axes[0])
        # autocorrelation_plot(s[1], ax=axes[1])
        # autocorrelation_plot(s[2], ax=axes[2])

    def quantize(self, bitdepth=16):
        pass


# ----------------------------------------------------------------------------------------------------------------------
# if __name__ == '__main__':
#     s = Signal()
#     print(a(2))
#     print(a(3))
#     print(a(4))
#     print(a(5))
#
#     print(a(2, max=2.0))
#     print(a(3, max=2.0))
#     print(a(4, max=2.0))
#     print(a(5, max=2.0))
#
#     print(a(2, min=1.0, max=2.0))
#     print(a(3, min=1.0, max=2.0))
#     print(a(4, min=1.0, max=2.0))
#     print(a(5, min=1.0, max=2.0))
