from gymnasium.spaces import Space
from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
from typing import Any, Iterable, Mapping

import collections.abc
import typing
from collections import OrderedDict
from typing import Any, Callable
from .sspspace import SSP
from numpy.typing import NDArray
import numpy as np

from .ssp_box import SSPBox
from .ssp_discrete import SSPDiscrete
from .ssp_sequence import SSPSequence

class SSPDict(Space[typing.Dict[str, Any]], typing.Mapping[str, Space[Any]]):
    r"""A dictionary of SSP :class:`Space` instances.
    Example:
        >>> observation_space = SSPDict({"position": SSPBox(-1, 1, shape_in=2, shape_out=256),
                                         "color": SSPDiscrete(3, shape_out=256)}, seed=42)
    """

    def __init__(
        self,
        spaces = None, #: None | dict[str, Space] | Sequence[tuple[str, Space]]
        static_spaces = None,
        seed = None, #: dict | int | np.random.Generator | None 
        **spaces_kwargs
    ):
        r"""Constructor of :class:`SSPDict`.

        """
        if isinstance(spaces, collections.abc.Mapping) and not isinstance(
            spaces, OrderedDict
        ):
            try:
                spaces = OrderedDict(sorted(spaces.items()))
            except TypeError:
                # Incomparable types (e.g. `int` vs. `str`, or user-defined types) found.
                # The keys remain in the insertion order.
                spaces = OrderedDict(spaces.items())
        elif isinstance(spaces, Sequence):
            spaces = OrderedDict(spaces)
        elif spaces is None:
            spaces = OrderedDict()
        else:
            assert isinstance(
                spaces, OrderedDict
            ), f"Unexpected Dict space input, expecting dict, OrderedDict or Sequence, actual type: {type(spaces)}"

        # Add kwargs to spaces to allow both dictionary and keywords to be used
        for key, space in spaces_kwargs.items():
            if key not in spaces:
                spaces[key] = space
            else:
                raise ValueError(
                    f"Dict space keyword '{key}' already exists in the spaces dictionary."
                )

        self.spaces: dict[str, Space[Any]] = spaces
        self.static_spaces: dict[str, Space[Any]] = static_spaces
        for key, space in self.spaces.items():
            assert (isinstance( space, SSPBox) | isinstance( space, SSPDiscrete) | isinstance( space, SSPSequence)
            ), f"Dict space element is not an instance of SSP Space: key='{key}', space={space}"

        def _encode(x):
            raise ValueError(
                f"Enocding has not been defined, must first call define_encode."
            )
            
        def _decode(x):
            raise ValueError(
                f"Enocding has not been defined, must first call define_encode."
            )

        self._encode = _encode
        self._decode = _decode
        self._map_to_dict = lambda x: x
        self._map_from_dict = lambda x: x
        # None for shape and dtype, since it'll require special handling
        super().__init__(None, None, seed)

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return all(space.is_np_flattenable for space in self.spaces.values())
    
    def encode(self,x):
        return self._encode(x)
    
    def decode(self,x):
        return self._decode(x)
    
    def set_map_to_dict(self, func: Callable):
        self._map_to_dict = func
        
    def set_map_from_dict(self, func: Callable):
        self._map_from_dict = func
        
    def set_encode(self, func: Callable):
        def _encode(x):
            x = self._map_to_dict(x)
            for key in self.spaces.keys():
                x[key] = SSP(self.spaces[key].encode(x[key]))
            return func(x, self.static_spaces)
        self._encode = _encode
        
    def set_decode(self, func: Callable):
        def _decode(x):
            return self._map_from_dict(func(x, self.spaces, self.static_spaces))
        self._decode = _decode
        

    def seed(self, seed = None):
        """Seed the PRNG of this space and all subspaces.

        Depending on the type of seed, the subspaces will be seeded differently

        * ``None`` - All the subspaces will use a random initial seed
        * ``Int`` - The integer is used to seed the :class:`Dict` space that is used to generate seed values for each of the subspaces. Warning, this does not guarantee unique seeds for all of the subspaces.
        * ``Dict`` - Using all the keys in the seed dictionary, the values are used to seed the subspaces. This allows the seeding of multiple composite subspaces (``Dict["space": Dict[...], ...]`` with ``{"space": {...}, ...}``).

        Args:
            seed: An optional list of ints or int to seed the (sub-)spaces.
        """
        seeds: list[int] = []

        if isinstance(seed, dict):
            assert (
                seed.keys() == self.spaces.keys()
            ), f"The seed keys: {seed.keys()} are not identical to space keys: {self.spaces.keys()}"
            for key in seed.keys():
                seeds += self.spaces[key].seed(seed[key])
        elif isinstance(seed, int):
            seeds = super().seed(seed)
            # Using `np.int32` will mean that the same key occurring is extremely low, even for large subspaces
            subseeds = self.np_random.integers(
                np.iinfo(np.int32).max, size=len(self.spaces)
            )
            for subspace, subseed in zip(self.spaces.values(), subseeds):
                seeds += subspace.seed(int(subseed))
        elif seed is None:
            for space in self.spaces.values():
                seeds += space.seed(None)
        else:
            raise TypeError(
                f"Expected seed type: dict, int or None, actual type: {type(seed)}"
            )

        return seeds

    def sample(self, mask = None):
        """Generates a single random sample from this space.

        The sample is an ordered dictionary of independent samples from the constituent spaces.

        Args:
            mask: An optional mask for each of the subspaces, expects the same keys as the space

        Returns:
            A dictionary with the same key and sampled values from :attr:`self.spaces`
        """
        if mask is not None:
            assert isinstance(
                mask, dict
            ), f"Expects mask to be a dict, actual type: {type(mask)}"
            assert (
                mask.keys() == self.spaces.keys()
            ), f"Expect mask keys to be same as space keys, mask keys: {mask.keys()}, space keys: {self.spaces.keys()}"
            return OrderedDict(
                [(k, space.sample(mask[k])) for k, space in self.spaces.items()]
            )
        sample = OrderedDict([(k, space.sample()) for k, space in self.spaces.items()])
        try:
            return self._encode(sample)
        except:
            return sample

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, dict) and x.keys() == self.spaces.keys():
            return all(x[key] in self.spaces[key] for key in self.spaces.keys())
        return False

    def __getitem__(self, key: str) -> Space[Any]:
        """Get the space that is associated to `key`."""
        return self.spaces[key]

    def __setitem__(self, key: str, value: Space[Any]):
        """Set the space that is associated to `key`."""
        assert isinstance(
            value, Space
        ), f"Trying to set {key} to Dict space with value that is not a gymnasium space, actual type: {type(value)}"
        self.spaces[key] = value

    def __iter__(self):
        """Iterator through the keys of the subspaces."""
        yield from self.spaces

    def __len__(self) -> int:
        """Gives the number of simpler spaces that make up the `Dict` space."""
        return len(self.spaces)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return (
            "Dict(" + ", ".join([f"{k!r}: {s}" for k, s in self.spaces.items()]) + ")"
        )

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return (
            isinstance(other, Dict)
            # Comparison of `OrderedDict`s is order-sensitive
            and self.spaces == other.spaces  # OrderedDict.__eq__
        )

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # serialize as dict-repr of vectors
        return {
            key: space.to_jsonable([sample[key] for sample in sample_n])
            for key, space in self.spaces.items()
        }

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        dict_of_list: dict[str, list[Any]] = {
            key: space.from_jsonable(sample_n[key])
            for key, space in self.spaces.items()
        }

        n_elements = len(next(iter(dict_of_list.values())))
        result = [
            OrderedDict({key: value[n] for key, value in dict_of_list.items()})
            for n in range(n_elements)
        ]
        return result