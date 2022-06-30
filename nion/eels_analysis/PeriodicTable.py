from __future__ import annotations

# standard libraries
import fractions
import json
import operator
import pkgutil
import typing

# third party libraries
# None

# local libraries
# None

class Singleton(type):
    def __init__(cls, name: str, bases: typing.Tuple[typing.Type[typing.Any], ...], d: typing.Dict[str, typing.Any]) -> None:
        super(Singleton, cls).__init__(name, bases, d)
        cls.instance: typing.Any = None

    def __call__(cls, *args: typing.Any, **kw: typing.Any) -> typing.Any:
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance


# see https://en.wikipedia.org/wiki/Electron_shell
# shell_number === principle quantum number n, or K, L, M, N, O, P, etc.
# subshell_index === EELS notation subshell
# K = 1s, L1 = 2s, L2 = 2p1/2, L3 = 2p3/2, M1 = 3s, M2 = 3p1/2, M3 = 3p3/2, M4 = 3d1/2, M5 = 3d3/2, etc.
class ElectronShell:
    def __init__(self, atomic_number: int, shell_number: int, subshell_index: int):
        self.atomic_number = atomic_number
        self.shell_number = shell_number
        self.subshell_index = subshell_index

    def __str__(self) -> str:
        return "{}-{}".format(PeriodicTable().element_symbol(self.atomic_number), self.get_shell_str_in_eels_notation(True))

    def to_long_str(self, include_subshell: bool = False) -> str:
        binding_energy_ev = PeriodicTable().nominal_binding_energy_ev(self)
        eels_shell_str = "{}-{}".format(PeriodicTable().element_symbol(self.atomic_number), self.get_shell_str_in_eels_notation(include_subshell))
        return "{}{}".format(eels_shell_str, " {:.1f} eV".format(binding_energy_ev) if binding_energy_ev is not None else str())

    def get_shell_str_in_eels_notation(self, include_subshell: bool = False) -> str:
        shell_str = chr(ord('K') + self.shell_number - 1)
        if (shell_str != 'K') and include_subshell:
            shell_str += str(self.subshell_index)
        return shell_str

    @classmethod
    def from_eels_notation(cls, atomic_number: int, eels_shell: str) -> ElectronShell:
        shell_number = ord(eels_shell[0].upper()) - ord('K') + 1
        if eels_shell == "K":
            return ElectronShell(atomic_number, shell_number, 1)
        subshell_index = int(eels_shell[1:])
        return ElectronShell(atomic_number, shell_number, subshell_index)

    @property
    def azimuthal_quantum_number(self) -> int:
        aqn_table = (0, 0, 1, 1, 2, 2, 3, 3, 4, 4)  # first value is a dummy
        assert self.subshell_index >= 1
        return aqn_table[self.subshell_index]

    @property
    def subshell_label(self) -> str:
        # azimuthal_quantum_number runs from 0 (s-states) to 4 (f-states).
        subshell_labels = ('s', 'p', 'd', 'f', 'g', 'h', 'i', 'j')
        return subshell_labels[self.azimuthal_quantum_number]

    @property
    def spin_fraction(self) -> fractions.Fraction:
        # subshell_index runs from 1 (s-states) to 7 (f-states).
        spins = (0, 1, 1, 3, 3, 5, 5, 7, 7, 9)  # first value is a dummy
        assert self.subshell_index >= 1
        return fractions.Fraction(spins[self.subshell_index], 2)


class PeriodicTable(metaclass=Singleton):
    def __init__(self) -> None:
        json_data = pkgutil.get_data(__name__, "resources/edges.json")
        assert json_data is not None
        self.__edge_data: typing.Sequence[typing.Mapping[str, typing.Any]] = json.loads(json_data)

    def element_symbol(self, atomic_number: int) -> str:
        for edge_data_item in self.__edge_data:
            if edge_data_item.get("z", 0) == atomic_number:
                return typing.cast(str, edge_data_item.get("symbol"))
        raise IndexError()

    def nominal_binding_energy_ev(self, electron_shell: ElectronShell) -> float:
        for edge_data_item in self.__edge_data:
            if edge_data_item.get("z", 0) == electron_shell.atomic_number:
                return typing.cast(float, edge_data_item.get("edges", dict()).get(electron_shell.get_shell_str_in_eels_notation(True)))
        raise IndexError()

    def get_elements_list(self) -> typing.Tuple[typing.Tuple[int, str], ...]:
        """Return a list of tuples: atomic number, atomic symbol."""
        return tuple((typing.cast(int, edge_data_item.get("z")), typing.cast(str, edge_data_item.get("symbol"))) for edge_data_item in self.__edge_data)

    def get_edges_list(self, atomic_number: int) -> typing.Sequence[typing.Tuple[ElectronShell, str]]:
        """Return a list of tuples: electron shell (lowest energy within shell number), edge name (without subshell)."""
        for edge_data_item in self.__edge_data:
            if edge_data_item.get("z", 0) == atomic_number:
                edge_dict = edge_data_item.get("edges", dict())
                edge_map: typing.Dict[int, typing.Tuple[ElectronShell, float]] = dict()
                for eels_shell, energy in edge_dict.items():
                    electron_shell = ElectronShell.from_eels_notation(atomic_number, eels_shell)
                    base_electron_shell_energy = edge_map[electron_shell.shell_number][1] if electron_shell.shell_number in edge_map else 1E9
                    if energy < base_electron_shell_energy:
                        edge_map[electron_shell.shell_number] = (electron_shell, energy)
                return list((edge_map[key][0], edge_map[key][0].to_long_str()) for key in sorted(edge_map.keys()))
        return list()

    def find_edges_in_energy_interval(self, energy_interval_ev: typing.Tuple[float, float]) -> typing.List[ElectronShell]:
        """Return list of electron shells found within energy interval, sorted by distance from center."""
        edges = list()  # typing.List[typing.Tuple[float, ElectronShell]]
        energy_interval_center_ev = (energy_interval_ev[0] + energy_interval_ev[1]) * 0.5
        for edge_data_item in self.__edge_data:
            atomic_number = edge_data_item.get("z", 0)
            edge_dict = edge_data_item.get("edges", dict())
            # find lowest energy edge within each shell
            edge_map: typing.Dict[int, typing.Tuple[ElectronShell, float]] = dict()
            for eels_shell, energy in edge_dict.items():
                electron_shell = ElectronShell.from_eels_notation(atomic_number, eels_shell)
                base_electron_shell_energy = edge_map[electron_shell.shell_number][
                    1] if electron_shell.shell_number in edge_map else 1E9
                if energy < base_electron_shell_energy:
                    edge_map[electron_shell.shell_number] = (electron_shell, energy)
            for electron_shell, energy in edge_map.values():
                if energy_interval_ev[0] <= energy <= energy_interval_ev[1]:
                    edges.append((abs(energy_interval_center_ev - energy), electron_shell))
        edges.sort(key=operator.itemgetter(0))
        return [edge[1] for edge in edges]



# print(ElectronShell.from_eels_notation(6, "M4"))
# print(ElectronShell.from_eels_notation(6, "M4").get_shell_str_in_eels_notation(True))
# print(PeriodicTable().nominal_binding_energy_ev(ElectronShell.from_eels_notation(35, "M4")))
# print([es.to_long_str() for es in PeriodicTable().find_edges_in_energy_interval((1833, 1933))])
