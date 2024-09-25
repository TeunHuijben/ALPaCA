import numpy as np
from vendored.smuthi.fields import blocksize, multi_to_single_index, angular_frequency

class FieldExpansion:
    """Base class for field expansions."""

    def __init__(self):
        self.validity_conditions = []

    def valid(self, x, y, z):
        """Test if points are in definition range of the expansion. 
        Abstract method to be overwritten in child classes.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            numpy.ndarray of bool datatype indicating if points are inside 
            definition domain.
        """
        ret = np.ones(x.shape, dtype=bool)
        for check in self.validity_conditions:
            ret = np.logical_and(ret, check(x, y, z))
        return ret

    def diverging(self, x, y, z):
        """Test if points are in domain where expansion could diverge. Virtual 
        method to be overwritten in child 
        classes.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            numpy.ndarray of bool datatype indicating if points are inside 
            divergence domain.
        """
        pass

    def electric_field(self, x, y, z):
        """Evaluate electric field. Virtual method to be overwritten in child 
        classes.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            Tuple of (E_x, E_y, E_z) numpy.ndarray objects with the Cartesian 
            coordinates of complex electric field.
        """
        pass
    
    def magnetic_field(self, x, y, z, vacuum_wavelength):
        """Evaluate magnetic field. Virtual method to be overwritten in child 
        classes.
        
        Args:
            x (numpy.ndarray):          x-coordinates of query points
            y (numpy.ndarray):          y-coordinates of query points
            z (numpy.ndarray):          z-coordinates of query points
            vacuum_wavelength (float):  Vacuum wavelength in length units
         
        Returns:
            Tuple of (H_x, H_y, H_z) numpy.ndarray objects with the Cartesian 
            coordinates of complex magnetic field.
        """
        pass


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

    def valid(self, x, y, z):
        """Test if points are in definition range of the expansion.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            numpy.ndarray of bool datatype indicating if points are inside 
            definition domain.
        """
        vld = np.logical_and(z >= self.lower_z, z < self.upper_z)
        return np.logical_and(vld, FieldExpansion.valid(self, x, y, z))

    def diverging(self, x, y, z):
        """Test if points are in domain where expansion could diverge.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            numpy.ndarray of bool datatype indicating if points are inside 
            divergence domain.
        """
        r = np.sqrt((x - self.reference_point[0])**2 + (y - self.reference_point[1])**2
                    + (z - self.reference_point[2])**2)
        if self.kind == 'regular':
            return r >= self.outer_r
        if self.kind == 'outgoing':
            return r <= self.inner_r
        else:
            return None

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
    
    def electric_field(self, x, y, z):
        """Evaluate electric field.
        
        Args:
            x (numpy.ndarray):    x-coordinates of query points
            y (numpy.ndarray):    y-coordinates of query points
            z (numpy.ndarray):    z-coordinates of query points
         
        Returns:
            Tuple of (E_x, E_y, E_z) numpy.ndarray objects with the Cartesian 
            coordinates of complex electric field.
        """
        x = np.array(x, ndmin=1)
        y = np.array(y, ndmin=1)
        z = np.array(z, ndmin=1)

        xr = x[self.valid(x, y, z)] - self.reference_point[0]
        yr = y[self.valid(x, y, z)] - self.reference_point[1]
        zr = z[self.valid(x, y, z)] - self.reference_point[2]
        ex = np.zeros(x.shape, dtype=complex)
        ey = np.zeros(x.shape, dtype=complex)
        ez = np.zeros(x.shape, dtype=complex)
        for tau in range(2):
            for m in range(-self.m_max, self.m_max + 1):
                for l in range(max(1, abs(m)), self.l_max + 1):
                    b = self.coefficients_tlm(tau, l, m)
                    if self.kind == 'regular':
                        Nx, Ny, Nz = vwf.spherical_vector_wave_function(xr, yr, zr, self.k, 1, tau, l, m)
                    elif self.kind == 'outgoing':
                        Nx, Ny, Nz = vwf.spherical_vector_wave_function(xr, yr, zr, self.k, 3, tau, l, m)
                    ex[self.valid(x, y, z)] += b * Nx
                    ey[self.valid(x, y, z)] += b * Ny
                    ez[self.valid(x, y, z)] += b * Nz
        return ex, ey, ez
    
    def magnetic_field(self, x, y, z, vacuum_wavelength):
        """Evaluate magnetic field.
        
        Args:
            x (numpy.ndarray):          x-coordinates of query points
            y (numpy.ndarray):          y-coordinates of query points
            z (numpy.ndarray):          z-coordinates of query points
            vacuum_wavelength (float):  Vacuum wavelength in length units
         
        Returns:
            Tuple of (H_x, H_y, H_z) numpy.ndarray objects with the Cartesian
            coordinates of complex electric field.
        """
        omega = angular_frequency(vacuum_wavelength)
        
        x = np.array(x, ndmin=1)
        y = np.array(y, ndmin=1)
        z = np.array(z, ndmin=1)

        xr = x[self.valid(x, y, z)] - self.reference_point[0]
        yr = y[self.valid(x, y, z)] - self.reference_point[1]
        zr = z[self.valid(x, y, z)] - self.reference_point[2]
        hx = np.zeros(x.shape, dtype=complex)
        hy = np.zeros(x.shape, dtype=complex)
        hz = np.zeros(x.shape, dtype=complex)
        for tau in range(2):
            for m in range(-self.m_max, self.m_max + 1):
                for l in range(max(1, abs(m)), self.l_max + 1):
                    b = self.coefficients_tlm(1-tau, l, m)
                    if self.kind == 'regular':
                        Nx, Ny, Nz = vwf.spherical_vector_wave_function(xr, yr, zr, self.k, 1, tau, l, m)
                    elif self.kind == 'outgoing':
                        Nx, Ny, Nz = vwf.spherical_vector_wave_function(xr, yr, zr, self.k, 3, tau, l, m)
                    hx[self.valid(x, y, z)] += b * Nx
                    hy[self.valid(x, y, z)] += b * Ny
                    hz[self.valid(x, y, z)] += b * Nz
        
        hx = - 1j * self.k / omega * hx
        hy = - 1j * self.k / omega * hy
        hz = - 1j * self.k / omega * hz      
        
        return hx, hy, hz

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