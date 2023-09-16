from gymnasium.spaces.space import Space
from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
from typing import Any, Iterable, Mapping

from .sspspace import *
from numpy.typing import NDArray
import numpy as np

class SSPDiscrete(Space[np.ndarray]):
    r"""SSP represenation of a (possibly unbounded) box in :math:`\mathbb{R}^n`.
    """

    def __init__(
        self,
        n: int,
        shape_out: int = None,
        dtype: Type = np.float32,
        seed: Optional[Union[int, np.random.Generator]] = None,
        ssp_space = None,
        start: int = 0
    ):
        r"""Constructor of :class:`SSPDiscrete` space.

        This will construct the space :math:`\{\text{start}, ..., \text{start} + n - 1\}`.

        Args:
            n (int): The number of elements of this space.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the ``Dict`` space.
            start (int): The smallest element of this space.
        """
        assert (
            dtype is not None
        ), "SSPDiscrete dtype must be explicitly provided, cannot be None."
        assert np.issubdtype(
            type(n), np.integer
        ), f"Expects `n` to be an integer, actual dtype: {type(n)}"
        assert n > 0, "n (counts) have to be positive"
        assert np.issubdtype(
            type(start), np.integer
        ), f"Expects `start` to be an integer, actual type: {type(start)}"

        self.n = np.int64(n)
        self.start = np.int64(start)
        self.dtype = np.dtype(dtype)
        self.shape_out = shape_out
        self.shape_in = (1,)

        if ssp_space is None:
            self.ssp_space = SPSpace(
                                self.n,
                                ssp_dim=self.shape_out, 
                                seed=seed)
        else:
            self.ssp_space = ssp_space
        
        super().__init__((self.shape_out, ), self.dtype, seed)
        

    @property
    def shape(self) -> Tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self.shape_out

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def encode(self, x):
        return self.ssp_space.encode(x - self.start)
    
    def decode(self, ssp):
        return self.ssp_space.decode(ssp).item() + self.start

    def sample(self, mask = None) -> np.int64:
        """Generates a single random sample from this space.

        A sample will be chosen uniformly at random with the mask if provided

        Args:
            mask: An optional mask for if an action can be selected.
                Expected `np.ndarray` of shape `(n,)` and dtype `np.int8` where `1` represents valid actions and `0` invalid / infeasible actions.
                If there are no possible actions (i.e. `np.all(mask == 0)`) then `space.start` will be returned.

        Returns:
            A sampled integer from the space
        """
        if mask is not None:
            assert isinstance(
                mask, np.ndarray
            ), f"The expected type of the mask is np.ndarray, actual type: {type(mask)}"
            assert (
                mask.dtype == np.int8
            ), f"The expected dtype of the mask is np.int8, actual dtype: {mask.dtype}"
            assert mask.shape == (
                self.n,
            ), f"The expected shape of the mask is {(self.n,)}, actual shape: {mask.shape}"
            valid_action_mask = mask == 1
            assert np.all(
                np.logical_or(mask == 0, valid_action_mask)
            ), f"All values of a mask should be 0 or 1, actual values: {mask}"
            if np.any(valid_action_mask):
                return self.encode(self.start + self.np_random.choice(
                    np.where(valid_action_mask)[0]
                ))
            else:
                return self.encode(self.start)

        return self.encode(self.start + self.np_random.integers(self.n)).reshape(-1)
        
    
    def samples(self, n_samples):
        r"""Generates many random samples inside the SSPBox.
        """
        return self.encode(self.start + self.np_random.integers(self.n, size=n_samples))

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return bool(np.any([np.allclose(self.ssp_space.vectors[j,:], x) for j in range(self.n)]))

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n: Sequence[Union[float, int]]) -> List[np.ndarray]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self) -> str:
        """A string representation of this space.

        The representation will include bounds, shape and dtype.
        If a bound is uniform, only the corresponding scalar will be given to avoid redundant and ugly strings.

        Returns:
            A representation of the space
        """
        return f"SSPDiscrete({self.n}, {self.shape_out}, {self.dtype})"

    def __eq__(self, other) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype or ssp_space equivalence."""
        return (
            isinstance(other, SSPDiscrete)
            and (self.n == other.n)
            and (self.shape_out == other.shape_out)
            and (self.seed == other.seed)
            and (self.seed is not None)
            and (other.seed is not None)
        )

    def __setstate__(self, state: Dict):
        """Used when loading a pickled space.

        This method has to be implemented explicitly to allow for loading of legacy states.

        Args:
            state: The new state
        """
        # Don't mutate the original state
        state = dict(state)

        # Allow for loading of legacy states.
        # See https://github.com/openai/gym/pull/2470
        if "start" not in state:
            state["start"] = np.int64(0)

        super().__setstate__(state)
            
def _short_repr(arr: NDArray[Any]) -> str:
    """Create a shortened string representation of a numpy array.

    If arr is a multiple of the all-ones vector, return a string representation of the multiplier.
    Otherwise, return a string representation of the entire array.

    Args:
        arr: The array to represent

    Returns:
        A short representation of the array
    """
    if arr.size != 0 and np.min(arr) == np.max(arr):
        return str(np.min(arr))
    return str(arr)


def is_float_integer(var: Any) -> bool:
    """Checks if a variable is an integer or float."""
    return np.issubdtype(type(var), np.integer) or np.issubdtype(type(var), np.floating)