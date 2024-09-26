import numpy as np
from vendored.smuthi.fields import blocksize, multi_to_single_index
from vendored.smuthi.fields import k_z as smuthi_k_z

class FieldExpansion:
    """Base class for field expansions."""

    def __init__(self):
        self.validity_conditions = []

class SphericalWaveExpansion(FieldExpansion):
    r"""A class to manage spherical wave expansions of the form

    .. math::
        \mathbf{E}(\mathbf{r}) = \sum_{\tau=1}^2 \sum_{l=1}^\infty \sum_{m=-l}^l a_{\tau l m} 
        \mathbf{\Psi}^{(\nu)}_{\tau l m}(\mathbf{r} - \mathbf{r}_i)

    for :math:`\mathbf{r}` located in a layer defined by 
    :math:`z\in [z_{min}, z_{max}]`
    and where :math:`\mathbf{\Psi}^{(\nu)}_{\tau l m}` are the SVWFs, see 
    :meth:`smuthi.vector_wave_functions.spherical_vector_wave_function`.

    Internally, the expansion coefficients :math:`a_{\tau l m}` are stored as a 
    1-dimensional array running over a multi index :math:`n` subsumming over 
    the SVWF indices :math:`(\tau,l,m)`. The 
    mapping from the SVWF indices to the multi
    index is organized by the function :meth:`multi_to_single_index`.
    
    Args:
        k (float):    wavenumber in layer where expansion is valid
        l_max (int):  maximal multipole degree :math:`l_\mathrm{max}\geq 1` 
                      where to truncate the expansion. 
        m_max (int):  maximal multipole order :math:`0 \leq m_\mathrm{max} \leq l_\mathrm{max}`
                      where to truncate the 
                      expansion.
        kind (str):   'regular' for :math:`\nu=1` or 'outgoing' for :math:`\nu=3`
        reference_point (list or tuple):  [x, y, z]-coordinates of point relative 
                                          to which the spherical waves are 
                                          considered (e.g., particle center).
        lower_z (float):   the expansion is valid on and above that z-coordinate
        upper_z (float):   the expansion is valid below that z-coordinate
        inner_r (float):   radius inside which the expansion diverges 
                           (e.g. circumscribing sphere of particle)
        outer_r (float):   radius outside which the expansion diverges

    Attributes:
        coefficients (numpy ndarray): expansion coefficients :math:`a_{\tau l m}` ordered by multi index :math:`n`                          
    """

    def __init__(self, k, l_max, m_max=None, kind=None, reference_point=None, lower_z=-np.inf, upper_z=np.inf,
                 inner_r=0, outer_r=np.inf):
        FieldExpansion.__init__(self)
        self.k = k
        self.l_max = l_max
        if m_max is not None:
            self.m_max = m_max
        else:
            self.m_max = l_max
        self.coefficients = np.zeros(blocksize(self.l_max, self.m_max), dtype=complex)
        self.kind = kind  # 'regular' or 'outgoing'
        self.reference_point = reference_point
        self.lower_z = lower_z
        self.upper_z = upper_z
        self.inner_r = inner_r
        self.outer_r = outer_r

    def coefficients_tlm(self, tau, l, m):
        """SWE coefficient for given (tau, l, m)

        Args:
            tau (int):  SVWF polarization (0 for spherical TE, 1 for spherical TM)
            l (int):    SVWF degree
            m (int):    SVWF order

        Returns:
            SWE coefficient
        """
        n = multi_to_single_index(tau, l, m, self.l_max, self.m_max)
        return self.coefficients[n]

    def compatible(self, other):
        """Check if two spherical wave expansions are compatible in the sense 
        that they can be added coefficient-wise

        Args:
            other (FieldExpansion):  expansion object to add to this object

        Returns:
            bool (true if compatible, false else)
        """
        return (type(other).__name__ == "SphericalWaveExpansion" 
                and self.k == other.k 
                and self.l_max == other.l_max
                and self.m_max == other.m_max 
                and self.kind == other.kind
                and np.array_equal(self.reference_point, other.reference_point))

    def __add__(self, other):
        """Addition of expansion objects (by coefficients).
        
        Args:
            other (SphericalWaveExpansion):  expansion object to add to this object
        
        Returns:
            SphericalWaveExpansion object as the sum of this expansion and the other
        """
        # todo: allow different l_max
        if not self.compatible(other):
            raise ValueError('SphericalWaveExpansions are inconsistent.')
        swe_sum = SphericalWaveExpansion(k=self.k, l_max=self.l_max, m_max=self.m_max, kind=self.kind,
                                         reference_point=self.reference_point, inner_r=max(self.inner_r, other.inner_r),
                                         outer_r=min(self.outer_r, other.outer_r),
                                         lower_z=max(self.lower_z, other.lower_z),
                                         upper_z=min(self.upper_z, other.upper_z))
        swe_sum.coefficients = self.coefficients + other.coefficients
        return swe_sum

class PlaneWaveExpansion(FieldExpansion):
    r"""A class to manage plane wave expansions of the form

    .. math::
        \mathbf{E}(\mathbf{r}) = \sum_{j=1}^2 \iint \mathrm{d}^2\mathbf{k}_\parallel \, g_{j}(\kappa, \alpha)
        \mathbf{\Phi}^\pm_j(\kappa, \alpha; \mathbf{r} - \mathbf{r}_i)

    for :math:`\mathbf{r}` located in a layer defined by :math:`z\in [z_{min}, z_{max}]`
    and :math:`\mathrm{d}^2\mathbf{k}_\parallel = \kappa\,\mathrm{d}\alpha\,\mathrm{d}\kappa`. 
    
    The double integral runs over :math:`\alpha\in[0, 2\pi]` and :math:`\kappa\in[0,\kappa_\mathrm{max}]`. 
    Further, :math:`\mathbf{\Phi}^\pm_j` are the PVWFs, see :meth:`plane_vector_wave_function`.

    Internally, the expansion coefficients :math:`g_{ij}^\pm(\kappa, \alpha)` 
    are stored as a 3-dimensional numpy ndarray.
    
    If the attributes k_parallel and azimuthal_angles have only a single entry, 
    a discrete distribution is assumed:

    .. math::
        g_{j}^\pm(\kappa, \alpha) \sim \delta^2(\mathbf{k}_\parallel - \mathbf{k}_{\parallel, 0})

    .. todo: update attributes doc

    Args:
        k (float):                          wavenumber in layer where expansion is valid
        k_parallel (numpy ndarray):         array of in-plane wavenumbers (can be float or complex)
        azimuthal_angles (numpy ndarray):   :math:`\alpha`, from 0 to :math:`2\pi`
        kind (str):                         'upgoing' for :math:`g^+` and 'downgoing' for :math:`g^-` type
                                            expansions 
        reference_point (list or tuple):    [x, y, z]-coordinates of point relative to which the plane waves are 
                                            defined.
        lower_z (float):                    the expansion is valid on and above that z-coordinate
        upper_z (float):                    the expansion is valid below that z-coordinate
        

    Attributes:
        coefficients (numpy ndarray): coefficients[j, k, l] contains 
        :math:`g^\pm_{j}(\kappa_{k}, \alpha_{l})`
    """
    def __init__(self, k, k_parallel, azimuthal_angles, kind=None, reference_point=None, lower_z=-np.inf,
                 upper_z=np.inf):
        FieldExpansion.__init__(self)
        self.k = k
        self.k_parallel = np.array(k_parallel, ndmin=1)
        self.azimuthal_angles = np.array(azimuthal_angles, ndmin=1)
        self.kind = kind  # 'upgoing' or 'downgoing'
        self.reference_point = reference_point
        self.lower_z = lower_z
        self.upper_z = upper_z

        # The coefficients :math:`g^\pm_{j}(\kappa,\alpha) are represented as a 3-dimensional numpy.ndarray.
        # The indices are:
        # - polarization (0=TE, 1=TM)
        # - index of the kappa dimension
        # - index of the alpha dimension
        self.coefficients = np.zeros((2, len(self.k_parallel), len(self.azimuthal_angles)), dtype=complex)

    def k_parallel_grid(self):
        """Meshgrid of n_effective with respect to azimuthal_angles"""
        kp_grid, _ = np.meshgrid(self.k_parallel, self.azimuthal_angles, indexing='ij')
        return kp_grid

    def azimuthal_angle_grid(self):
        """Meshgrid of azimuthal_angles with respect to n_effective"""
        _, a_grid = np.meshgrid(self.k_parallel, self.azimuthal_angles, indexing='ij')
        return a_grid

    def k_z(self):
        if self.kind == 'upgoing':
            kz = smuthi_k_z(k_parallel=self.k_parallel, k=self.k)
        elif self.kind == 'downgoing':
            kz = -smuthi_k_z(k_parallel=self.k_parallel, k=self.k)
        else:
            raise ValueError('pwe kind undefined')
        return kz

    def k_z_grid(self):
        if self.kind == 'upgoing':
            kz = smuthi_k_z(k_parallel=self.k_parallel_grid(), k=self.k)
        elif self.kind == 'downgoing':
            kz = -smuthi_k_z(k_parallel=self.k_parallel_grid(), k=self.k)
        else:
            raise ValueError('pwe type undefined')
        return kz

    def compatible(self, other):
        """Check if two plane wave expansions are compatible in the sense that 
        they can be added coefficient-wise

        Args:
            other (FieldExpansion):  expansion object to add to this object

        Returns:
            bool (true if compatible, false else)
        """
        return (type(other).__name__=="PlaneWaveExpansion" and np.isclose(self.k, other.k)
                and all(np.isclose(self.k_parallel, other.k_parallel))
                and all(np.isclose(self.azimuthal_angles, other.azimuthal_angles)) and self.kind == other.kind
                and np.array_equal(self.reference_point, other.reference_point))

    def __add__(self, other):
        if not self.compatible(other):
            raise ValueError('Plane wave expansion are inconsistent.')
        pwe_sum = PlaneWaveExpansion(k=self.k, k_parallel=self.k_parallel, azimuthal_angles=self.azimuthal_angles,
                                     kind=self.kind, reference_point=self.reference_point,
                                     lower_z=max(self.lower_z, other.lower_z),
                                     upper_z=min(self.upper_z, other.upper_z))
        pwe_sum.coefficients = self.coefficients + other.coefficients
        return pwe_sum