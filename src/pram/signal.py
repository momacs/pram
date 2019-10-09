import matplotlib.pyplot as plt
import numpy as np

from pandas.plotting               import autocorrelation_plot
from statsmodels.tsa.stattools     import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ----------------------------------------------------------------------------------------------------------------------
class Signal(object):
    '''
    Time series considered a signal.

    When considered over time, mass dynamics and other time series that PramPy output can be seen as signals.  Signal
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

    def __init__(self, S=None, names=None):
        if S is not None and not isinstance(S, np.ndarray):
            raise ValueError('S needs to be an instance of ndarray.')

        self.S = S  # series
        self.names = names
        # self.iter_max = 0

        if self.S is not None:
            if self.names is None:
                self.names = [None]*len(self.S)

            if len(self.S) != len(self.names):
                raise ValueError('The number of series and names must be identical.')

            # self.iter_max = max([len(s) for s in self.S])

    def add_series(self, s, name=None):
        if self.S is None or len(self.S) == 0:
            self.S = s
            self.names = [name]
        else:
            self.S = np.append(self.S, s, axis=0)
            self.names.append(name)

    def plot_autocorr(self, size, filename=None, do_ret_plot=False, dpi=100):
        fig, axes = plt.subplots(len(self.S), 1, figsize=size, sharex=True, dpi=dpi)
        fig.subplots_adjust(hspace=0, wspace=0)
        # plt.suptitle('', fontweight='bold')
        # plt.xlabel('Time', fontweight='bold')

        for i in range(len(self.S)):
            ax = axes[i]
            autocorrelation_plot(self.S[i,1:], ax=ax)
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
