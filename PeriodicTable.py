# standard libraries
import collections

# third party libraries
# None

# local libraries
# None

class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance


ElectronShell = collections.namedtuple("ElectronShell", ['atomic_number', 'shell_number', 'subshell_index'])


class PeriodicTable(metaclass=Singleton):
    def __init__(self):
        pass

    def nominal_binding_energy_eV(self, electron_shell: ElectronShell) -> float:
        return 0.0
