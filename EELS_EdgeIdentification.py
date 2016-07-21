"""
    EELS Edge Identification

    A library of functions for identifying and getting information about EELS ionization edges.

"""

# standard libraries
import collections

ElectronShell = collections.namedtuple("ElectronShell", ['atomic_number', 'shell_number', 'subshell_index'])

class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance

class PeriodicTable(metaclass=Singleton):
    def __init__(self):
        pass

    def nominal_binding_energy_ev(self, electron_shell: ElectronShell) -> float:
        return 0.0


def candidate_edges(edge_onset_eV: float, max_offset_eV: float, primary_subshells_only: bool) -> list:
    """Return a list of edges near the specified edge onset energy.
    """
    pass

def edge_identity(edge_onset_eV: float, element_symbol: str, atomic_number: int) -> ElectronShell:
    pass

def electron_shell(element_symbol: str, atomic_number: int, shell_number: int, subshell_index: int) -> ElectronShell:
    pass

def element_edges(element_symbol: str, atomic_number: int, shell_number: int) -> list:
    pass

def nominal_edge_onset_eV(edge: ElectronShell) -> float:
    """Return the electron binding energy for the given edge.

    Return value is in eV.
    """
    pass