import numpy as np

def plane_vector_wave_function(x, y, z, kp, alpha, kz, pol):
    r"""Electric field components of plane wave (PVWF), see section 2.3.1 of
    [Egel 2018 diss].

    .. math::
        \mathbf{\Phi}_j = \exp ( \mathrm{i} \mathbf{k} \cdot \mathbf{r} ) \hat{ \mathbf{e} }_j

    with :math:`\hat{\mathbf{e}}_0` denoting the unit vector in azimuthal direction ('TE' or 's' polarization),
    and :math:`\hat{\mathbf{e}}_1` denoting the unit vector in polar direction ('TM' or 'p' polarization).

    The input arrays should have one of the following dimensions:

        - x,y,z: (N x 1) matrix
        - kp,alpha,kz: (1 x M) matrix
        - Ex, Ey, Ez: (M x N) matrix

    or

        - x,y,z: (M x N) matrix
        - kp,alpha,kz: scalar
        - Ex, Ey, Ez: (M x N) matrix

    Args:
        x (numpy.ndarray): x-coordinate of position where to test the field (length unit)
        y (numpy.ndarray): y-coordinate of position where to test the field
        z (numpy.ndarray): z-coordinate of position where to test the field
        kp (numpy.ndarray): parallel component of k-vector (inverse length unit)
        alpha (numpy.ndarray): azimthal angle of k-vector (rad)
        kz (numpy.ndarray): z-component of k-vector (inverse length unit)
        pol (int): Polarization (0=TE, 1=TM)

    Returns:
        - x-coordinate of PVWF electric field (numpy.ndarray)
        - y-coordinate of PVWF electric field (numpy.ndarray)
        - z-coordinate of PVWF electric field (numpy.ndarray)
    """
    k = np.sqrt(kp**2 + kz**2)
    kx = kp * np.cos(alpha)
    ky = kp * np.sin(alpha)

    scalar_wave = np.exp(1j * (kx * x + ky * y + kz * z))

    if pol == 0:
        Ex = -np.sin(alpha) * scalar_wave
        Ey = np.cos(alpha) * scalar_wave
        Ez = np.zeros_like(scalar_wave)
    elif pol == 1:
        Ex = np.cos(alpha) * kz / k * scalar_wave
        Ey = np.sin(alpha) * kz / k * scalar_wave
        Ez = -kp / k * scalar_wave
    else:
        raise ValueError('Polarization must be 0 (TE) or 1 (TM)')

    return Ex, Ey, Ez