import numpy as np
from vendored.smuthi.fields import blocksize, multi_to_single_index
from vendored.smuthi.math import wigner_D, legendre_normalized
from vendored.smuthi.expansion import PlaneWaveExpansion

def block_rotation_matrix_D_svwf(l_max, m_max, alpha, beta, gamma, wdsympy=False):
    """Rotation matrix for the rotation of SVWFs between the labratory 
    coordinate system (L) and a rotated coordinate system (R)
    
    Args:
        l_max (int):      Maximal multipole degree
        m_max (int):      Maximal multipole order
        alpha (float):    First Euler angle, rotation around z-axis, in rad
        beta (float):     Second Euler angle, rotation around y'-axis in rad
        gamma (float):    Third Euler angle, rotation around z''-axis in rad
        wdsympy (bool):   If True, Wigner-d-functions come from the sympy toolbox
        
    Returns:
        rotation matrix of dimension [blocksize, blocksize]
    """
    
    b_size = blocksize(l_max, m_max)
    rotation_matrix = np.zeros([b_size, b_size], dtype=complex)
    
    for l in range(l_max + 1):
        mstop = min(l, m_max)
        for m1 in range(-mstop, mstop + 1):
            for m2 in range(-mstop, mstop + 1):
                rotation_matrix_coefficient = wigner_D(l, m1, m2, alpha, beta, gamma, wdsympy)
                for tau in range(2):
                    n1 = multi_to_single_index(tau, l, m1, l_max, m_max)
                    n2 = multi_to_single_index(tau, l, m2, l_max, m_max)
                    rotation_matrix[n1, n2] = rotation_matrix_coefficient

    return rotation_matrix


def transformation_coefficients_vwf(tau, l, m, pol, kp=None, kz=None, pilm_list=None, taulm_list=None, dagger=False):
    r"""Transformation coefficients B to expand SVWF in PVWF and vice versa:

    .. math::
        B_{\tau l m,j}(x) = -\frac{1}{\mathrm{i}^{l+1}} \frac{1}{\sqrt{2l(l+1)}} (\mathrm{i} \delta_{j1} + \delta_{j2})
        (\delta_{\tau j} \tau_l^{|m|}(x) + (1-\delta_{\tau j} m \pi_l^{|m|}(x))

    For the definition of the :math:`\tau_l^m` and :math:`\pi_l^m` functions, see
    `A. Doicu, T. Wriedt, and Y. A. Eremin: "Light Scattering by Systems of Particles", Springer-Verlag, 2006
    <https://doi.org/10.1007/978-3-540-33697-6>`_
    
    Compare also section 2.3.3 of [Egel 2018 diss].
    
    Args:
        tau (int):          SVWF polarization, 0 for spherical TE, 1 for spherical TM
        l (int):            l=1,... SVWF multipole degree
        m (int):            m=-l,...,l SVWF multipole order
        pol (int):          PVWF polarization, 0 for TE, 1 for TM
        kp (numpy array):         PVWF in-plane wavenumbers
        kz (numpy array):         complex numpy-array: PVWF out-of-plane wavenumbers
        pilm_list (list):   2D list numpy-arrays: alternatively to kp and kz, pilm and taulm as generated with
                            legendre_normalized can directly be handed
        taulm_list (list):  2D list numpy-arrays: alternatively to kp and kz, pilm and taulm as generated with
                            legendre_normalized can directly be handed
        dagger (bool):      switch on when expanding PVWF in SVWF and off when expanding SVWF in PVWF

    Returns:
        Transformation coefficient as array (size like kp).
    """
    if pilm_list is None:
        k = np.sqrt(kp**2 + kz**2)
        ct = kz / k
        st = kp / k
        plm_list, pilm_list, taulm_list = legendre_normalized(ct, st, l)

    if tau == pol:
        sphfun = taulm_list[l, abs(m)]
    else:
        sphfun = m * pilm_list[l, abs(m)]

    if dagger:
        if pol == 0:
            prefac = -1 / (-1j) ** (l + 1) / np.sqrt(2 * l * (l + 1)) * (-1j)
        elif pol == 1:
            prefac = -1 / (-1j) ** (l + 1) / np.sqrt(2 * l * (l + 1)) * 1
        else:
            raise ValueError('pol must be 0 (TE) or 1 (TM)')
    else:
        if pol == 0:
            prefac = -1 / (1j) ** (l + 1) / np.sqrt(2 * l * (l + 1)) * (1j)
        elif pol ==1:
            prefac = -1 / (1j) ** (l + 1) / np.sqrt(2 * l * (l + 1)) * 1
        else:
            raise ValueError('pol must be 0 (TE) or 1 (TM)')

    B = prefac * sphfun
    return B

def swe_to_pwe_conversion(swe, k_parallel, azimuthal_angles, layer_system=None, layer_number=None,
                          layer_system_mediated=False,
                          only_l=None, only_m=None, only_pol=None, only_tau=None):
    """Convert SphericalWaveExpansion object to a PlaneWaveExpansion object.

    Args:
        swe (SphericalWaveExpansion):             Spherical wave expansion to be converted
        k_parallel (numpy array or str):          In-plane wavenumbers for the pwe object.
        azimuthal_angles (numpy array or str):    Azimuthal angles for the pwe object
        layer_system (smuthi.layers.LayerSystem): Stratified medium in which the origin of the SWE is located
        layer_number (int):                       Layer number in which the PWE should be valid.
        layer_system_mediated (bool):             If True, the PWE refers to the layer system response of the SWE, 
                                                  otherwise it is the direct transform.
        only_pol (int):  if set to 0 or 1, only this plane wave polarization (0=TE, 1=TM) is considered
        only_tau (int):  if set to 0 or 1, only this spherical vector wave polarization (0 — magnetic, 1 — electric) is
                         considered
        only_l (int):    if set to positive number, only this multipole degree is considered
        only_m (int):    if set to non-negative number, only this multipole order is considered

    Returns:
        Tuple of two PlaneWaveExpansion objects, first upgoing, second downgoing.
    """
    # todo: manage diverging swe
    k_parallel = np.array(k_parallel, ndmin=1)
    
    i_swe = layer_system.layer_number(swe.reference_point[2])
    if layer_number is None and not layer_system_mediated:
        layer_number = i_swe
    reference_point = [0, 0, layer_system.reference_z(i_swe)]
    lower_z_up = swe.reference_point[2]
    upper_z_up = layer_system.upper_zlimit(layer_number)
    pwe_up = PlaneWaveExpansion(k=swe.k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles, kind='upgoing',
                                    reference_point=reference_point, lower_z=lower_z_up, upper_z=upper_z_up)
    lower_z_down = layer_system.lower_zlimit(layer_number)
    upper_z_down = swe.reference_point[2]
    pwe_down = PlaneWaveExpansion(k=swe.k, k_parallel=k_parallel, azimuthal_angles=azimuthal_angles,
                                      kind='downgoing', reference_point=reference_point, lower_z=lower_z_down,
                                      upper_z=upper_z_down)

    agrid = pwe_up.azimuthal_angle_grid()
    kpgrid = pwe_up.k_parallel_grid()
    kx = kpgrid * np.cos(agrid)
    ky = kpgrid * np.sin(agrid)
    kz_up = pwe_up.k_z_grid()
    kz_down = pwe_down.k_z_grid()

    kzvec = pwe_up.k_z()

    kvec_up = np.array([kx, ky, kz_up])
    kvec_down = np.array([kx, ky, kz_down])
    rpwe_mn_rswe = np.array(reference_point) - np.array(swe.reference_point)

    # phase factor for the translation of the reference point from rvec_S to rvec_iS
    ejkrSiS_up = np.exp(1j * np.tensordot(kvec_up, rpwe_mn_rswe, axes=([0], [0])))
    ejkrSiS_down = np.exp(1j * np.tensordot(kvec_down, rpwe_mn_rswe, axes=([0], [0])))
    
    ct_up = pwe_up.k_z() / swe.k
    st_up = pwe_up.k_parallel / swe.k
    plm_list_up, pilm_list_up, taulm_list_up = legendre_normalized(ct_up, st_up, swe.l_max)

    ct_down = pwe_down.k_z() / swe.k
    st_down = pwe_down.k_parallel / swe.k
    plm_list_down, pilm_list_down, taulm_list_down = legendre_normalized(ct_down, st_down, swe.l_max)
    
    for m in range(-swe.m_max, swe.m_max + 1):
        if only_m is not None and m != only_m:
            continue
        eima = np.exp(1j * m * pwe_up.azimuthal_angles)  # indices: alpha_idx
        for pol in range(2):
            if only_pol is not None and pol != only_pol:
                continue
            dbB_up = np.zeros(len(k_parallel), dtype=complex)
            dbB_down = np.zeros(len(k_parallel), dtype=complex)
            for l in range(max(1, abs(m)), swe.l_max + 1):
                if only_l is not None and l != only_l:
                    continue
                for tau in range(2):
                    if only_tau is not None and tau != only_tau:
                        continue
                    dbB_up += swe.coefficients_tlm(tau, l, m) * transformation_coefficients_vwf(
                        tau, l, m, pol, pilm_list=pilm_list_up, taulm_list=taulm_list_up)
                    dbB_down += swe.coefficients_tlm(tau, l, m) * transformation_coefficients_vwf(
                        tau, l, m, pol, pilm_list=pilm_list_down, taulm_list=taulm_list_down)
            pwe_up.coefficients[pol, :, :] += dbB_up[:, None] * eima[None, :]
            pwe_down.coefficients[pol, :, :] += dbB_down[:, None] * eima[None, :]

    pwe_up.coefficients = pwe_up.coefficients / (2 * np.pi * kzvec[None, :, None] * swe.k) * ejkrSiS_up[None, :, :]
    pwe_down.coefficients = (pwe_down.coefficients / (2 * np.pi * kzvec[None, :, None] * swe.k)
                             * ejkrSiS_down[None, :, :])

    if layer_system_mediated:
        pwe_up, pwe_down = layer_system.response((pwe_up, pwe_down), i_swe, layer_number)

    return pwe_up, pwe_down