import nengo_spa as spa
import nengo
import nengolib
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt


from nengo_spa.semantic_pointer import SemanticPointer
from nengo.utils.compat import is_array, is_array_like, is_number
from nengo_spa.algebras.hrr_algebra import HrrAlgebra
from nengo_spa.ast.base import Fixed, infer_types, TypeCheckedBinaryOp
from nengo_spa.types import TAnyVocab, TScalar, TVocabulary

from nengo_spa.algebras.hrr_algebra import HrrAlgebra

from nengo.params import (
    NdarrayParam,
    FrozenObject,
)
from nengo.dists import Distribution, UniformHypersphere
from nengo.exceptions import ValidationError


# The SemanticPointer class, copied from nengo-spa, with fractional binding  via ``**`` added
class SemanticPointer(Fixed):
    """A Semantic Pointer, based on Holographic Reduced Representations.
    Operators are overloaded so that ``+`` and ``-`` are addition,
    ``*`` is circular convolution, ``**`` is fractional circular convolution,
    and ``~`` is the inversion operator.
    Parameters
    ----------
    data : array_like
        The vector constituting the Semantic Pointer.
    vocab : Vocabulary, optional
        Vocabulary that the Semantic Pointer is considered to be part of.
        Mutually exclusive with the *algebra* argument.
    algebra : AbstractAlgebra, optional
        Algebra used to perform vector symbolic operations on the Semantic
        Pointer. Defaults to `.CircularConvolutionAlgebra`. Mutually exclusive
        with the *vocab* argument.
    name : str, optional
        A name for the Semantic Pointer.
    Attributes
    ----------
    v : array_like
        The vector constituting the Semantic Pointer.
    algebra : AbstractAlgebra
        Algebra that defines the vector symbolic operations on this Semantic
        Pointer.
    vocab : Vocabulary or None
        The vocabulary the this Semantic Pointer is considered to be part of.
    name : str or None
        Name of the Semantic Pointer.
    """

    def __init__(self, data, vocab=None, algebra=None, name=None):
        super(SemanticPointer, self).__init__(
            TAnyVocab if vocab is None else TVocabulary(vocab))
        self.algebra = self._get_algebra(vocab, algebra)

        self.v = np.array(data, dtype=complex)
        if len(self.v.shape) != 1:
            raise ValidationError("'data' must be a vector", 'data', self)
        self.v.setflags(write=False)

        self.vocab = vocab
        self.name = name

    def _get_algebra(cls, vocab, algebra):
        if algebra is None:
            if vocab is None:
                algebra = HrrAlgebra()
            else:
                algebra = vocab.algebra
        elif vocab is not None and vocab.algebra is not algebra:
            raise ValueError(
                "vocab and algebra argument are mutually exclusive")
        return algebra

    def _get_unary_name(self, op):
        return "{}({})".format(op, self.name) if self.name else None

    def _get_method_name(self, method):
        return "({}).{}()".format(self.name, method) if self.name else None

    def _get_binary_name(self, other, op, swap=False):
        if isinstance(other, SemanticPointer):
            other_name = other.name
        else:
            other_name = str(other)
        self_name = self.name
        if self_name and other_name:
            if swap:
                self_name, other_name = other_name, self.name
            return "({}){}({})".format(self_name, op, other_name)
        else:
            return None

    def evaluate(self):
        return self

    def connect_to(self, sink, **kwargs):
        return nengo.Connection(self.construct(), sink, **kwargs)

    def construct(self):
        return nengo.Node(self.v, label=str(self).format(len(self)))

    def normalized(self):
        """Normalize the Semantic Pointer and return it as a new object.
        If the vector length is zero, the Semantic Pointer will be returned
        unchanged.
        The original object is not modified.
        """
        nrm = np.linalg.norm(self.v)
        if nrm <= 0.:
            nrm = 1.
        return SemanticPointer(
            self.v / nrm, vocab=self.vocab, algebra=self.algebra,
            name=self._get_method_name("normalized"))

    def unitary(self):
        """Make the Semantic Pointer unitary and return it as a new object.
        The original object is not modified.
        A unitary Semantic Pointer has the property that it does not change
        the length of Semantic Pointers it is bound with using circular
        convolution.
        """
        return SemanticPointer(
            self.algebra.make_unitary(self.v), vocab=self.vocab,
            algebra=self.algebra, name=self._get_method_name("unitary"))

    def copy(self):
        """Return another semantic pointer with the same data."""
        return SemanticPointer(
            data=self.v, vocab=self.vocab, algebra=self.algebra,
            name=self.name)

    def length(self):
        """Return the L2 norm of the vector."""
        return np.linalg.norm(self.v)

    def __len__(self):
        """Return the number of dimensions in the vector."""
        return len(self.v)

    def __str__(self):
        if self.name:
            return "SemanticPointer<{}>".format(self.name)
        else:
            return repr(self)

    def __repr__(self):
        return (
            "SemanticPointer({!r}, vocab={!r}, algebra={!r}, name={!r}".format(
                self.v, self.vocab, self.algebra, self.name))

    @TypeCheckedBinaryOp(Fixed)
    def __add__(self, other):
        return self._add(other, swap=False)

    @TypeCheckedBinaryOp(Fixed)
    def __radd__(self, other):
        return self._add(other, swap=True)

    def _add(self, other, swap=False):
        type_ = infer_types(self, other)
        vocab = None if type_ == TAnyVocab else type_.vocab
        if vocab is None:
            self._ensure_algebra_match(other)
        other_pointer = other.evaluate()
        a, b = self.v, other_pointer.v
        if swap:
            a, b = b, a
        return SemanticPointer(
            data=self.algebra.superpose(a, b), vocab=vocab,
            algebra=self.algebra,
            name=self._get_binary_name(other_pointer, "+", swap))

    def __neg__(self):
        return SemanticPointer(
            data=-self.v, vocab=self.vocab, algebra=self.algebra,
            name=self._get_unary_name("-"))

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        """Multiplication of two SemanticPointers is circular convolution.
        If multiplied by a scalar, we do normal multiplication.
        """
        return self._mul(other, swap=False)

    def __rmul__(self, other):
        """Multiplication of two SemanticPointers is circular convolution.
        If multiplied by a scalar, we do normal multiplication.
        """
        return self._mul(other, swap=True)

    def _mul(self, other, swap=False):
        if is_array(other):
            raise TypeError(
                "Multiplication of Semantic Pointers with arrays in not "
                "allowed.")
        elif is_number(other):
            return SemanticPointer(
                data=self.v * other, vocab=self.vocab, algebra=self.algebra,
                name=self._get_binary_name(other, "*", swap))
        elif isinstance(other, Fixed):
            if other.type == TScalar:
                return SemanticPointer(
                    data=self.v * other.evaluate(), vocab=self.vocab,
                    algebra=self.algebra,
                    name=self._get_binary_name(other, "*", swap))
            else:
                return self._bind(other, swap=swap)
        else:
            return NotImplemented

    def __invert__(self):
        """Return a reorganized vector that acts as an inverse for convolution.
        This reorganization turns circular convolution into circular
        correlation, meaning that ``A*B*~B`` is approximately ``A``.
        For the vector ``[1, 2, 3, 4, 5]``, the inverse is ``[1, 5, 4, 3, 2]``.
        """
        return SemanticPointer(
            data=self.algebra.invert(self.v), vocab=self.vocab,
            algebra=self.algebra, name=self._get_unary_name("~"))

    def bind(self, other):
        """Return the binding of two SemanticPointers."""
        return self._bind(other, swap=False)

    def rbind(self, other):
        """Return the binding of two SemanticPointers."""
        return self._bind(other, swap=True)

    def _bind(self, other, swap=False):
        type_ = infer_types(self, other)
        vocab = None if type_ == TAnyVocab else type_.vocab
        if vocab is None:
            self._ensure_algebra_match(other)
        other_pointer = other.evaluate()
        a, b = self.v, other_pointer.v
        if swap:
            a, b = b, a
        return SemanticPointer(
            data=self.algebra.bind(a, b), vocab=vocab, algebra=self.algebra,
            name=self._get_binary_name(other_pointer, "*", swap))

    def get_binding_matrix(self, swap_inputs=False):
        """Return the matrix that does a binding with this vector.
        This should be such that
        ``A*B == dot(A.get_binding_matrix(), B.v)``.
        """
        return self.algebra.get_binding_matrix(self.v, swap_inputs=swap_inputs)

    def dot(self, other):
        """Return the dot product of the two vectors."""
        if isinstance(other, Fixed):
            infer_types(self, other)
            other = other.evaluate().v
        if is_array_like(other):
            return np.vdot(self.v, other)
        else:
            return other.vdot(self)

    def __matmul__(self, other):
        return self.dot(other)

    def compare(self, other):
        """Return the similarity between two SemanticPointers.
        This is the normalized dot product, or (equivalently), the cosine of
        the angle between the two vectors.
        """
        if isinstance(other, SemanticPointer):
            infer_types(self, other)
            other = other.evaluate().v
        scale = np.linalg.norm(self.v) * np.linalg.norm(other)
        if scale == 0:
            return 0
        return np.dot(self.v, other) / scale

    def reinterpret(self, vocab):
        """Reinterpret the Semantic Pointer as part of vocabulary *vocab*.
        The *vocab* parameter can be set to *None* to clear the associated
        vocabulary and allow the *source* to be interpreted as part of the
        vocabulary of any Semantic Pointer it is combined with.
        """
        return SemanticPointer(self.v, vocab=vocab, name=self.name)

    def translate(self, vocab, populate=None, keys=None, solver=None):
        """Translate the Semantic Pointer to vocabulary *vocab*.
        The translation of a Semantic Pointer uses some form of projection to
        convert the Semantic Pointer to a Semantic Pointer of another
        vocabulary. By default the outer products of terms in the source and
        target vocabulary are used, but if *solver* is given, it is used to
        find a least squares solution for this projection.
        Parameters
        ----------
        vocab : Vocabulary
            Target vocabulary.
        populate : bool, optional
            Whether the target vocabulary should be populated with missing
            keys.  This is done by default, but with a warning. Set this
            explicitly to *True* or *False* to silence the warning or raise an
            error.
        keys : list, optional
            All keys to translate. If *None*, all keys in the source vocabulary
            will be translated.
        solver : nengo.Solver, optional
            If given, the solver will be used to solve the least squares
            problem to provide a better projection for the translation.
        """
        tr = self.vocab.transform_to(vocab, populate, solver)
        return SemanticPointer(
            np.dot(tr, self.evaluate().v), vocab=vocab, name=self.name)

    def distance(self, other):
        """Return a distance measure between the vectors.
        This is ``1-cos(angle)``, so that it is 0 when they are identical, and
        the distance gets larger as the vectors are farther apart.
        """
        return 1 - self.compare(other)

    def mse(self, other):
        """Return the mean-squared-error between two vectors."""
        if isinstance(other, SemanticPointer):
            infer_types(self, other)
            other = other.evaluate().v
        return np.sum((self.v - other)**2) / len(self.v)

    def _ensure_algebra_match(self, other):
        """Check the algebra of the *other*.
        If the *other* parameter is a `SemanticPointer` and uses a different
        algebra, a `TypeError` will be raised.
        """
        if isinstance(other, SemanticPointer):
            if self.algebra is not other.algebra:
                raise TypeError(
                    "Operation not supported for SemanticPointer with "
                    "different algebra.")
                
    def __pow__(self, other):
        """Exponentiation of a SemanticPointer is fractional binding."""
        if is_number(other):
            return self.fractional_bind(other)
        else:
            return NotImplemented
    
    def fractional_bind(self, other):
        """Return the fractional binding of a SemanticPointer."""
        type_ = infer_types(self)
        vocab = None if type_ == TAnyVocab else type_.vocab
        a, b = self.v, other
        return SemanticPointer(
            data=self.algebra.fractional_bind(a, b), vocab=vocab, algebra=self.algebra,
            name=self._get_binary_name(other, "**", False))


class Identity(SemanticPointer):
    """Identity element.
    Parameters
    ----------
    n_dimensions : int
        Dimensionality of the identity vector.
    vocab : Vocabulary, optional
        Vocabulary that the Semantic Pointer is considered to be part of.
        Mutually exclusive with the *algebra* argument.
    algebra : AbstractAlgebra, optional
        Algebra used to perform vector symbolic operations on the Semantic
        Pointer. Defaults to `.CircularConvolutionAlgebra`. Mutually exclusive
        with the *vocab* argument.
    """

    def __init__(self, n_dimensions, vocab=None, algebra=None):
        data = self._get_algebra(vocab, algebra).identity_element(n_dimensions)
        super(Identity, self).__init__(
            data, vocab=vocab, algebra=algebra, name="Identity")


class AbsorbingElement(SemanticPointer):
    r"""Absorbing element.
    If :math:`z` denotes the absorbing element, :math:`v \circledast z = c z`,
    where :math:`v` is a Semantic Pointer and :math:`c` is a real-valued
    scalar. Furthermore :math:`\|z\| = 1`.
    Parameters
    ----------
    n_dimensions : int
        Dimensionality of the identity vector.
    vocab : Vocabulary, optional
        Vocabulary that the Semantic Pointer is considered to be part of.
        Mutually exclusive with the *algebra* argument.
    algebra : AbstractAlgebra, optional
        Algebra used to perform vector symbolic operations on the Semantic
        Pointer. Defaults to `.CircularConvolutionAlgebra`. Mutually exclusive
        with the *vocab* argument.
    """
    def __init__(self, n_dimensions, vocab=None, algebra=None):
        data = self._get_algebra(vocab, algebra).absorbing_element(
            n_dimensions)
        super(AbsorbingElement, self).__init__(
            data, vocab=vocab, algebra=algebra, name="AbsorbingElement")


class Zero(SemanticPointer):
    """Zero element.
    Parameters
    ----------
    n_dimensions : int
        Dimensionality of the identity vector.
    vocab : Vocabulary, optional
        Vocabulary that the Semantic Pointer is considered to be part of.
        Mutually exclusive with the *algebra* argument.
    algebra : AbstractAlgebra, optional
        Algebra used to perform vector symbolic operations on the Semantic
        Pointer. Defaults to `.CircularConvolutionAlgebra`. Mutually exclusive
        with the *vocab* argument.
    """
    def __init__(self, n_dimensions, vocab=None, algebra=None):
        data = self._get_algebra(vocab, algebra).zero_element(n_dimensions)
        super(Zero, self).__init__(
            data, vocab=vocab, algebra=algebra, name="Zero")

# HrrAlgebra with fractional binding added
class HrrAlgebra(HrrAlgebra):
    def fractional_bind(self, A, b):
        """Fractional circular convolution.""" 
        if not is_number(b):
            raise ValueError("b must be a scalar.")
        return np.fft.ifft(np.fft.fft(A, axis=0)**b, axis=0)#.real
    
    def bind(self, a, b):
        n = len(a)
        if len(b) != n:
            raise ValueError("Inputs must have same length.")
        return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b), n=n)
        #return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=n)
        
        
def ssp(X, Y, x, y, alg=HrrAlgebra()):
    # Return a ssp
    if ((type(X) == SemanticPointer) & (type(Y) == SemanticPointer)):
        return (X**x) * (Y**y)
    else:
        return (SemanticPointer(data=X,algebra=alg)**x) * (SemanticPointer(data=Y,algebra=alg)**y)
    
def ssp_vectorized(basis, positions):
    # Given a matrix of basis vectors, d by n (d = dimension of semantic pointer basis vectors, n = number of basis 
    # vectors, and a matrix of positions, N by n (N = number of points)
    # Return a matrix of N ssp vectors
    # Assuming the circular convolution defn for fractional binding
    positions = positions.reshape(-1,basis.shape[1])
    S_list = np.zeros((basis.shape[0],positions.shape[0]))
    for i in np.arange(positions.shape[0]):
        S_list[:,i] = np.fft.ifft(np.prod(np.fft.fft(basis, axis=0)**positions[i,:], axis=1), axis=0)  
    return S_list
    
def similarity_values(basis, positions, position0 = None, S0 = None, S_list = None):
    if position0 is None:
        position0 = np.zeros(basis.shape[1])
    if S0 is None:
        S0 = ssp_vectorized(basis, position0)
    if S_list is None: 
        S_list = ssp_vectorized(basis, positions)
    sim_dots = S_list.T @ S0
    return(sim_dots, S_list)

def similarity_plot(X, Y, xs, ys, x=0, y=0, S_list = None, S0 = None, check_mark= False, **kwargs):
    # Heat plot of SSP similarity of x and y values of xs and ys
    # Input:
    #  X, Y - SSP basis vectors
    #  x, y - A single point to compare SSPs over the space with
    #  xs, ys - The x, y points to make the space tiling
    #  titleStr - (optional) Title of plot
    #  S_list - (optional) A list of the SSPs at all xs, ys tiled points (useful for high dim X,Y so that these do not 
    #           have to recomputed every time this function is called)
    #  S0 - (optional) The SSP representing the x, y point (useful if for some reason you want a similarity plot
    #       of tiled SSPs with a non-SSP vector or a SSP with a different basis)
    #  check_mark - (default True) Whether or not to put a black check mark at the x, y location
    xx,yy = np.meshgrid(xs,ys)
    basis = np.vstack([X.v, Y.v]).T
    positions = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
    position0 = np.array([x,y])
    sim_dots, S_list = similarity_values(basis,  positions, position0 = position0, S0 = S0, S_list = S_list)
    plt.pcolormesh(xx, yy, sim_dots.reshape(xx.shape), **kwargs)
    if check_mark:
        plt.plot(x,y, 'k+')
    return(sim_dots, S_list)
    
def add_item_pts(item_locations, items_markers, items_cols):
    # Add items to plot at locations with marker symbols and colors given
    for i in np.arange(item_locations.shape[0]):
        plt.scatter(item_locations[i,0],item_locations[i,1],
                marker=items_markers[i],s=60,c=items_cols[i],edgecolors='w')
        
def similarity_items_plot(M, Objs, X, Y, xs, ys, S_list = None, S0 = None, check_mark= False, **kwargs):
    # Unbind each object from memory and add together the results - will be a sum of approximate SSPs 
    # representing the location of each object - and plot heat map
    # Run add_item_pts after to get item positions marked
    xx,yy = np.meshgrid(xs,ys)
    basis = np.vstack([X.v, Y.v]).T
    positions = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
    position0 = np.array([0,0])
    
    sim_dots, S_list = similarity_values(basis,  positions, position0 = position0, S0 = M * ~Objs[0], S_list = S_list)
    for i in np.arange(1,len(Objs)):
        obj_dots, _ = similarity_values(basis,  positions, position0 = position0, S0 = M * ~Objs[i], S_list = S_list)
        sim_dots += obj_dots
    plt.pcolormesh(xx, yy, sim_dots.reshape(xx.shape), cmap='viridis')
    

def ssp_plane_basis(K):
    # Create the bases vectors X,Y as described in the paper with the wavevectors 
    # (k_i = (u_i,v_i)) given in a matrix K. To get hexganal patterns use 3 K vectors 120 degs apart
    # To get mulit-scales/orientation, give many such sets of 3 K vectors 
    # K is _ by 2 
    d = K.shape[0]
    FX = np.ones((d*2 + 1,), dtype="complex")
    FX[0:d] = np.exp(1.j*K[:,0])
    FX[-d:] = np.flip(np.conj(FX[0:d]))
    FX = np.fft.ifftshift(FX)
    FY = np.ones((d*2 + 1,), dtype="complex")
    FY[0:d] = np.exp(1.j*K[:,1])
    FY[-d:] = np.flip(np.conj(FY[0:d]))
    FY = np.fft.ifftshift(FY)
    
    X = SemanticPointer(data=np.fft.ifft(FX), algebra=HrrAlgebra())
    Y = SemanticPointer(data=np.fft.ifft(FY), algebra=HrrAlgebra())
    return X, Y

def ssp_hex_basis(n_rotates,n_scales,scale_min=0.8, scale_max=3):
    # Create bases vectors X,Y consisting of mulitple sets of hexagonal bases
    K_hex = np.array([[0,1], [np.sqrt(3)/2,-0.5], [-np.sqrt(3)/2,-0.5]])

    scales = np.linspace(scale_min,scale_max,n_scales)
    K_scales = np.vstack([K_hex*i for i in scales])
    thetas = np.arange(0,n_rotates)*np.pi/(3*n_rotates)
    R_mats = np.stack([np.stack([np.cos(thetas), -np.sin(thetas)],axis=1),
           np.stack([np.sin(thetas), np.cos(thetas)], axis=1)], axis=1)
    K_rotates = (R_mats @ K_hex.T).transpose(1,2,0).T.reshape(-1,2)
    K_scale_rotates = (R_mats @ K_scales.T).transpose(1,2,0).T.reshape(-1,2)
    X, Y = ssp_plane_basis(K_scale_rotates)
    return X, Y, K_scale_rotates
    
def ssp_weighted_plane_basis(K,W):
    # The above but plane waves aren't just all summed. Instead there's a weighted sum - can get distortions in patterns
    # or make place cells more refined this way
    d = K.shape[0]
    FX = np.ones((d*2 + 1,), dtype="complex")
    FX[0:d] = W*np.exp(1.j*K[:,0])
    FX[-d:] = np.flip(np.conj(FX[0:d]))
    FX = np.fft.ifftshift(FX)
    FY = np.ones((d*2 + 1,), dtype="complex")
    FY[0:d] = W*np.exp(1.j*K[:,1])
    FY[-d:] = np.flip(np.conj(FY[0:d]))
    FY = np.fft.ifftshift(FY)
    
    X = SemanticPointer(data=np.fft.ifft(FX), algebra=HrrAlgebra())
    Y = SemanticPointer(data=np.fft.ifft(FY), algebra=HrrAlgebra())
    return X, Y

    
def planewave_mat(K, xx, yy, x0=0, y0=0):
    # Sum all plane waves to get inference pattern.
    # If you make SSPs with basis vectors from ssp_plane_basis(K) and call 
    # sim_dots, _ = similarity_plot(X, Y, xs, ys, x0, y0) 
    # then sim_dots should be the same as whats returned here. This is a check/quicker way to try out patterns
    mat = np.zeros(xx.shape)
    for i in np.arange(K.shape[0]):
        plane_wave = np.exp(1.j*(K[i,0]*(xx-x0) + K[i,1]*(yy-y0)))
        mat += (plane_wave + np.conj(plane_wave)).real
    return mat

def weighted_planewave_mat(K, xx, yy, W, x0=0, y0=0):
    # Above but give plane waves different weighting in the sum
    mat = np.zeros(xx.shape)
    for i in np.arange(K.shape[0]):
        plane_wave = W[i]*exp(1.j*(K[i,0]*(xx-x0) + K[i,1]*(yy-y0)))
        mat += (plane_wave + np.conj(plane_wave)).real
    return mat


def get_sub_FourierSSP(n, N, sublen=3):
    # Return a matrix, \bar{A}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \bar{A}_n F{S_{total}} = F{S_n}
    # i.e. pick out the sub vector in the Fourier domain
    tot_len = 2*sublen*N + 1
    FA = np.zeros((2*sublen + 1, tot_len))
    FA[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FA[sublen, sublen*N] = 1
    FA[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FA

def get_sub_SSP(n,N,sublen=3):
    # Return a matrix, A_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # A_n S_{total} = S_n
    # i.e. pick out the sub vector in the time domain
    tot_len = 2*sublen*N + 1
    FA = get_sub_FourierSSP(n,N,sublen=sublen)
    W = np.fft.fft(np.eye(tot_len))
    invW = np.fft.ifft(np.eye(2*sublen + 1))
    A = invW @ np.fft.ifftshift(FA) @ W
    return A.real

def proj_sub_FourierSSP(n,N,sublen=3):
    # Return a matrix, \bar{B}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n \bar{B}_n F{S_{n}} = F{S_{total}}
    # i.e. project the sub vector in the Fourier domain such that summing all such projections gives the full vector in Fourier domain
    tot_len = 2*sublen*N + 1
    FB = np.zeros((2*sublen + 1, tot_len))
    FB[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FB[sublen, sublen*N] = 1/N # all sub vectors have a "1" zero freq term so scale it so full vector will have 1 
    FB[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FB.T

def proj_sub_SSP(n,N,sublen=3):
    # Return a matrix, B_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n B_n S_{n} = S_{total}
    # i.e. project the sub vector in the time domain such that summing all such projections gives the full vector
    tot_len = 2*sublen*N + 1
    FB = proj_sub_FourierSSP(n,N,sublen=sublen)
    invW = np.fft.ifft(np.eye(tot_len))
    W = np.fft.fft(np.eye(2*sublen + 1))
    B = invW @ np.fft.ifftshift(FB) @ W
    return B.real


class UniformSSPs(Distribution):
# Get SSPs representing positions uniformly distributed. For setting encoders   
    X =  NdarrayParam("X", shape="*")
    Y =  NdarrayParam("Y", shape="*")
    alg = FrozenObject()

    def __init__(self, X, Y, alg = HrrAlgebra(), radius = 1):
        super().__init__()
        self.radius = radius
        if ((type(X) == SemanticPointer) & (type(Y) == SemanticPointer)):
            self.X = X.v
            self.Y = Y.v
            self.alg = X.algebra
        else:
            self.X = X
            self.Y = Y
            self.alg = alg

    def sample(self, n, d=None, rng=np.random):
            
        unif_dist = UniformHypersphere()
        xy = unif_dist.sample(n, 2)
        
        samples= np.zeros((n,d))
        for i in np.arange(n):
            samples[i,:] = ssp(self.X, self.Y, xy[i,0], xy[i,1], alg=self.alg).v.real
    
        return samples.real*self.radius
    
class ScatteredSSPs(Distribution):
# Get SSPs representing positions randomly distributed. For setting encoders     
    X =  NdarrayParam("X", shape="*")
    Y =  NdarrayParam("Y", shape="*")
    alg = FrozenObject()

    def __init__(self, X, Y, alg = HrrAlgebra(), radius = 1):
        super().__init__()
        self.radius = radius
        if ((type(X) == SemanticPointer) & (type(Y) == SemanticPointer)):
            self.X = X.v
            self.Y = Y.v
            self.alg = X.algebra
        else:
            self.X = X
            self.Y = Y
            self.alg = alg

    def sample(self, n, d=None, rng=np.random):
            
        unif_dist = nengolib.stats.ScatteredHypersphere(True)
        xy = unif_dist.sample(n, 2)
        
        samples= np.zeros((n,d))
        for i in np.arange(n):
            samples[i,:] = ssp(self.X, self.Y, xy[i,0], xy[i,1], alg=self.alg).v.real
        
        return samples.real*self.radius

    
def get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp):
    """
    Precompute spatial semantic pointers for every location in the linspace
    Used to quickly compute heat maps by a simple vectorized dot product (matrix multiplication)
    """
    if x_axis_sp.__class__.__name__ == 'SemanticPointer':
        dim = len(x_axis_sp.v)
    else:
        dim = len(x_axis_sp)
        x_axis_sp = spa.SemanticPointer(data=x_axis_sp)
        y_axis_sp = spa.SemanticPointer(data=y_axis_sp)

    vectors = np.zeros((len(xs), len(ys), dim))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = ssp(x_axis_sp, y_axis_sp, x, y)
            vectors[i, j, :] = p.v

    return vectors

    
# From github.com/ctn-waterloo/cogsci2019-ssp
def loc_match(sp, heatmap_vectors, xs, ys):
    if sp.__class__.__name__ == 'SemanticPointer':
        vs = np.tensordot(sp.v, heatmap_vectors, axes=([0], [2]))
    else:
        vs = np.tensordot(sp, heatmap_vectors, axes=([0], [2]))

    xy = np.unravel_index(vs.argmax(), vs.shape)

    x = xs[xy[0]]
    y = ys[xy[1]]
    return x,y, vs[xy]

    # Not similar enough to anything, so count as incorrect
   # if vs[xy] < sim_threshold:
   #     return 0

    # If within threshold of the correct location, count as correct
   # if (x-coord[0])**2 + (y-coord[1])**2 < distance_threshold**2:
   #     return 1
   # else:
    #    return 0
    

    
def loc_dist(sp, heatmap_vectors, coord, xs, ys, sim_threshold=0.5):
    if sp.__class__.__name__ == 'SemanticPointer':
        vs = np.tensordot(sp.v, heatmap_vectors, axes=([0], [2]))
    else:
        vs = np.tensordot(sp, heatmap_vectors, axes=([0], [2]))

    xy = np.unravel_index(vs.argmax(), vs.shape)

    x = xs[xy[0]]
    y = ys[xy[1]]

    # Not similar enough to anything, so count as incorrect
   # if vs[xy] < sim_threshold:
    #    return 10

    return np.sqrt((x-coord[0])**2 + (y-coord[1])**2 )

def make_good_unitary(D, eps=1e-3, rng=np.random):
    a = rng.rand((D - 1) // 2)
    sign = rng.choice((-1, +1), len(a))
    phi = sign * np.pi * (eps + a * (1 - 2 * eps))
    assert np.all(np.abs(phi) >= np.pi * eps)
    assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fv = np.zeros(D, dtype='complex64')
    fv[0] = 1
    fv[1:(D + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
    fv[-1:D // 2:-1] = np.conj(fv[1:(D + 1) // 2])
    if D % 2 == 0:
        fv[D // 2] = 1

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    # assert np.allclose(v.imag, 0, atol=1e-5)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return SemanticPointer(v)

# Path generating fnuctions 
def circle_rw(n,r,x0,y0,sigma):
    pts = np.zeros((n,2))
    pts[0,:]=np.array([x0,y0])
    for i in np.arange(1,n):
        newpt = sigma*np.random.randn(2) 
        if (np.linalg.norm(pts[i-1,:]+newpt)>r):
            pts[i,:]=pts[i-1,:]-newpt
        else:
            pts[i,:]=pts[i-1,:]+newpt
            
    return(pts)

def random_path(radius, n_steps, dims, fac):
    walk = np.zeros((n_steps,dims))
    pt_old = np.zeros((1,dims))
    for i in np.arange(n_steps):
        walk[i,:] = pt_old
        step_vec = (np.random.rand(dims)-0.5)*fac
        pt_new = np.maximum(np.minimum(pt_old+step_vec, radius), -radius)
        pt_old = pt_new
    return walk

def generate_signal(T,dt,dims = 1, rms=0.5,limit=10, seed=1):
    np.random.seed(seed)             
    N = int(T/dt)
    dw = 2*np.pi/T
    
    # Don't get samples for outside limit, those coeffs will stay zero
    num_samples = max(1,min(N//2, int(2*np.pi*limit/dw)))
    
    x_freq = np.zeros((N,dims), dtype=complex)
    x_freq[0,:] = np.random.randn(dims) #zero-frequency coeffient
    x_freq[1:num_samples+1,:] = np.random.randn(num_samples,dims) + 1j*np.random.randn(num_samples,dims) #postive-frequency coeffients
    x_freq[-num_samples:,:] += np.flip(x_freq[1:num_samples+1,:].conjugate(),axis=0)  #negative-frequency coeffients
      
    x_time = np.fft.ifft(x_freq,n=N,axis=0)
    x_time = x_time.real # it is real, but in case of numerical error, make sure
    rescale = rms/np.sqrt(dt*np.sum(x_time**2)/T)
    x_time = rescale*x_time
    x_freq = rescale*x_freq
    
    x_freq = np.fft.fftshift(x_freq)    
    return(x_time,x_freq)
 

# Used for saving certain figures. 
# https://brushingupscience.wordpress.com/2017/05/09/vector-and-raster-in-one-with-matplotlib/
def rasterize_and_save(fname, rasterize_list=None, fig=None, dpi=None,
                       savefig_kw={}):
    """Save a figure with raster and vector components
    This function lets you specify which objects to rasterize at the export
    stage, rather than within each plotting call. Rasterizing certain
    components of a complex figure can significantly reduce file size.
    Inputs
    ------
    fname : str
        Output filename with extension
    rasterize_list : list (or object)
        List of objects to rasterize (or a single object to rasterize)
    fig : matplotlib figure object
        Defaults to current figure
    dpi : int
        Resolution (dots per inch) for rasterizing
    savefig_kw : dict
        Extra keywords to pass to matplotlib.pyplot.savefig
    If rasterize_list is not specified, then all contour, pcolor, and
    collects objects (e.g., ``scatter, fill_between`` etc) will be
    rasterized
    Note: does not work correctly with round=True in Basemap
    Example
    -------
    Rasterize the contour, pcolor, and scatter plots, but not the line
    >>> import matplotlib.pyplot as plt
    >>> from numpy.random import random
    >>> X, Y, Z = random((9, 9)), random((9, 9)), random((9, 9))
    >>> fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    >>> cax1 = ax1.contourf(Z)
    >>> cax2 = ax2.scatter(X, Y, s=Z)
    >>> cax3 = ax3.pcolormesh(Z)
    >>> cax4 = ax4.plot(Z[:, 0])
    >>> rasterize_list = [cax1, cax2, cax3]
    >>> rasterize_and_save('out.svg', rasterize_list, fig=fig, dpi=300)
    """

    # Behave like pyplot and act on current figure if no figure is specified
    fig = plt.gcf() if fig is None else fig

    # Need to set_rasterization_zorder in order for rasterizing to work
    zorder = -5  # Somewhat arbitrary, just ensuring less than 0

    if rasterize_list is None:
        # Have a guess at stuff that should be rasterised
        types_to_raster = ['QuadMesh', 'Contour', 'collections']
        rasterize_list = []

        print("""
        No rasterize_list specified, so the following objects will
        be rasterized: """)
        # Get all axes, and then get objects within axes
        for ax in fig.get_axes():
            for item in ax.get_children():
                if any(x in str(item) for x in types_to_raster):
                    rasterize_list.append(item)
        print('\n'.join([str(x) for x in rasterize_list]))
    else:
        # Allow rasterize_list to be input as an object to rasterize
        if type(rasterize_list) != list:
            rasterize_list = [rasterize_list]

    for item in rasterize_list:

        # Whether or not plot is a contour plot is important
        is_contour = (isinstance(item, matplotlib.contour.QuadContourSet) or
                      isinstance(item, matplotlib.tri.TriContourSet))

        # Whether or not collection of lines
        # This is commented as we seldom want to rasterize lines
        # is_lines = isinstance(item, matplotlib.collections.LineCollection)

        # Whether or not current item is list of patches
        all_patch_types = tuple(
            x[1] for x in getmembers(matplotlib.patches, isclass))
        try:
            is_patch_list = isinstance(item[0], all_patch_types)
        except TypeError:
            is_patch_list = False

        # Convert to rasterized mode and then change zorder properties
        if is_contour:
            curr_ax = item.ax.axes
            curr_ax.set_rasterization_zorder(zorder)
            # For contour plots, need to set each part of the contour
            # collection individually
            for contour_level in item.collections:
                contour_level.set_zorder(zorder - 1)
                contour_level.set_rasterized(True)
        elif is_patch_list:
            # For list of patches, need to set zorder for each patch
            for patch in item:
                curr_ax = patch.axes
                curr_ax.set_rasterization_zorder(zorder)
                patch.set_zorder(zorder - 1)
                patch.set_rasterized(True)
        else:
            # For all other objects, we can just do it all at once
            curr_ax = item.axes
            curr_ax.set_rasterization_zorder(zorder)
            item.set_rasterized(True)
            item.set_zorder(zorder - 1)

    # dpi is a savefig keyword argument, but treat it as special since it is
    # important to this function
    if dpi is not None:
        savefig_kw['dpi'] = dpi

    # Save resulting figure
    fig.savefig(fname, **savefig_kw)
    
    
    
class PathIntegrator(nengo.Network):
    def __init__(self, n_neurons, n_gridcells, scale_fac=1.0, basis=None,xy_rad=10, **kwargs):
        kwargs.setdefault("label", "PathIntegrator")
        super().__init__(**kwargs)
        
        
        if basis is None:
            K_hex = np.array([[0,1], [np.sqrt(3)/2,-0.5], [-np.sqrt(3)/2,-0.5]])
            n_scales = 5
            scales = np.linspace(0.5,2.5,n_scales)
            K_scales = np.vstack([K_hex*i for i in scales])
            n_rotates = 5
            thetas = np.arange(0,n_rotates)*np.pi/(3*n_rotates)
            R_mats = np.stack([np.stack([np.cos(thetas), -np.sin(thetas)],axis=1),
                       np.stack([np.sin(thetas), np.cos(thetas)], axis=1)], axis=1)
            K_rotates = (R_mats @ K_hex.T).transpose(1,2,0).T.reshape(-1,2)
            K_scale_rotates = (R_mats @ K_scales.T).transpose(1,2,0).T.reshape(-1,2)
            N = n_scales*n_rotates
            X, Y = ssp_plane_basis(K_scale_rotates)
            myK = K_scale_rotates
            d = X.v.shape[0]
        else:
            X = basis[0]
            Y = basis[1]
            d = X.v.shape[0]
            N = (d - 1)//6
            myK = np.vstack([np.angle(np.fft.fftshift(np.fft.fft(X.v)))[0:d//2],
                             np.angle(np.fft.fftshift(np.fft.fft(Y.v)))[0:d//2]]).T
            
        n_oscs = d//2
        real_ids = np.arange(1,n_oscs*3,3)
        imag_ids = np.arange(2,n_oscs*3,3)
        S_ids = np.zeros(n_oscs*2 + 1, dtype=int)
        S_ids[0:d//2] = real_ids
        S_ids[d//2:(n_oscs*2)] = imag_ids
        S_ids[-1] = n_oscs*3 
        i_S_ids = np.argsort(S_ids)
        
        G_pos_dist = nengolib.stats.Rd()
        G_pos = G_pos_dist.sample(n_gridcells,2)*xy_rad
        G_sorts = np.hstack([np.arange(N), np.random.randint(0, N - 1, size = n_gridcells - N)])
        G_encoders = np.zeros((n_gridcells,d))
        for i in np.arange(n_gridcells):
            sub_mat = get_sub_SSP(G_sorts[i],N)
            proj_mat = proj_sub_SSP(G_sorts[i],N)
            Xi = SemanticPointer(data = sub_mat @ X.v)
            Yi = SemanticPointer(data = sub_mat @ Y.v)
            G_encoders[i,:] = N * proj_mat @ ((Xi**G_pos[i,0])*(Yi**G_pos[i,1])).v
        n_eval_pts = nengo.utils.builder.default_n_eval_points(n_gridcells, d)
        unif_dist = nengolib.stats.ScatteredHypersphere(True)
        eval_xy = xy_rad*unif_dist.sample(n_eval_pts, 2)
        eval_pts = ssp_vectorized(np.vstack([X.v, Y.v]).T, eval_xy).real.T

        taus = 0.1*np.ones(n_oscs)
            

        with self:
            self.input_vel = nengo.Node(size_in=2, label="input_vel")
            self.input_initial_SSP = nengo.Node(size_in=d, label="input_initial_SSP")
            self.output = nengo.Node(size_in=d, label="output")
            
            self.velocity = nengo.Ensemble(n_neurons, dimensions=2,label='velocity')
            zero_freq_term = nengo.Node([1,0,0])
            
            self.osc = nengo.networks.EnsembleArray(n_neurons, n_oscs + 1, 
                                                    ens_dimensions = 3,radius=np.sqrt(3), label="osc")
            self.osc.output.output = lambda t, x: x # a hack
            self.grid_cells = nengo.Ensemble(n_gridcells, dimensions=d, encoders = G_encoders,
                                        radius=np.sqrt(2), label="grid_cells")
            
            def feedback(x, tau):
                w = x[0]/scale_fac
                r = np.maximum(np.sqrt(x[1]**2 + x[2]**2), 1e-5)
                dx1 = x[1]*(1-r**2)/r - x[2]*w 
                dx2 = x[2]*(1-r**2)/r + x[1]*w 
                return 0, tau*dx1 + x[1], tau*dx2 + x[2]
            
            to_SSP = self.get_to_SSP_mat(d)
            #i_to_SSP = self.get_from_SSP_mat(d)

            nengo.Connection(self.input_vel, self.velocity, transform = scale_fac)
            for i in np.arange(n_oscs):
                nengo.Connection(self.velocity, self.osc.ea_ensembles[i][0], transform = myK[i,:].reshape(1,-1),  
                                 synapse=taus[i])
                nengo.Connection(stim[i], osc.ea_ensembles[i][1]) #initialize
                nengo.Connection(stim[i + d//2], osc.ea_ensembles[i][2]) #initialize
                nengo.Connection(self.osc.ea_ensembles[i], self.osc.ea_ensembles[i], 
                                 function= lambda x: feedback(x, taus[i]), 
                                 synapse=taus[i])

                #S_back_mat = i_to_SSP[i_S_ids[2*i:(2*i+2)],:]
                #nengo.Connection(self.grid_cells, self.osc.ea_ensembles[i][1:], transform=S_back_mat, synapse=taus[i])
                
            nengo.Connection(zero_freq_term, self.osc.ea_ensembles[-1])
            nengo.Connection(self.osc.output[S_ids], self.grid_cells, transform = to_SSP, synapse=taus[0]) 

            nengo.Connection(self.input_initial_SSP, self.grid_cells)
            
            nengo.Connection(self.grid_cells, self.output)
            
            

    
            
            
    def get_to_SSP_mat(self,D):
        W = np.fft.ifft(np.eye(D))
        W1 = W.real @ np.fft.ifftshift(np.eye(D),axes=0)
        W2 = W.imag @ np.fft.ifftshift(np.eye(D),axes=0)
        shiftmat1 = np.vstack([np.eye(D//2), np.zeros((1,D//2)), np.flip(np.eye(D//2), axis=0)])
        shiftmat2 = np.vstack([np.eye(D//2), np.zeros((1,D//2)), -np.flip(np.eye(D//2), axis=0)])
        shiftmat = np.vstack([ np.hstack([shiftmat1, np.zeros(shiftmat2.shape)]),
                              np.hstack([np.zeros(shiftmat2.shape), shiftmat2])])
        shiftmat = np.hstack([shiftmat, np.zeros((shiftmat.shape[0],1))])
        shiftmat[D//2,-1] = 1
        tr = np.hstack([W1, -W2]) @ shiftmat 

        return tr

    def get_from_SSP_mat(self,D):
        W = np.fft.fft(np.eye(D))
        W1 = np.fft.fftshift(np.eye(D),axes=0) @ W.real 
        W2 = np.fft.fftshift(np.eye(D),axes=0) @ W.imag 
        shiftmat1 = np.hstack([np.eye(D//2), np.zeros((D//2, 2*(D//2) + D//2 + 2))])
        shiftmat2 = np.hstack([np.zeros((D//2, 2*(D//2) + 1)), np.eye(D//2), np.zeros((D//2, D//2 + 1))])
        shiftmat = np.vstack([ shiftmat1,shiftmat2])
        tr = shiftmat @ np.vstack([W1, W2]) 
        return tr