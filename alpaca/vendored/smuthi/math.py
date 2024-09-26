import numpy as np

def factorial(n):
    """Return factorial.

    Args:
        n (int): Argument (non-negative)

    Returns:
        Factorial of n
    """
    assert type(n) == int and n >= 0
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
    
def wigner_d(l, m, m_prime, beta, wdsympy=False):
    """Computation of Wigner-d-functions for the rotation of a T-matrix

    Args:
        l (int):          Degree :math:`l` (1, ..., lmax)
        m (int):          Order :math:`m` (-min(l,mmax),...,min(l,mmax))
        m_prime (int):    Order :math:`m_prime` (-min(l,mmax),...,min(l,mmax))
        beta (float):     Second Euler angle in rad
        wdsympy (bool):   If True, Wigner-d-functions come from the sympy toolbox 
        
    Returns:
        real value of Wigner-d-function
    """
    wig_d = np.zeros(l + 1, dtype=complex)
    
    if wdsympy == False:
    
        if beta < 0:
            aa = m
            bb = m_prime
            m = bb
            m_prime = aa
           
        if m == 0 and m_prime == 0:
            for nn in range(1, l + 1):
                P_nn = np.polynomial.legendre.Legendre.basis(nn)
                wig_d[nn] = P_nn(np.cos(beta))          
        else:
            # recursion formulation (Mishchenko, Scattering, Absorption and Emission of Light by small Particles, p.365 (B.22 - B.24))
            l_min = max(abs(m), abs(m_prime))
            wig_d[l_min - 1] = 0
            if m_prime >= m:
                zeta = 1
            else:
                zeta = (-1) ** (m - m_prime)
        
            wig_d[l_min] = (zeta * 2.0 ** (-l_min) * (factorial(2 * l_min) / (factorial(abs(m - m_prime)) 
                                                                              * factorial(abs(m + m_prime)))) ** 0.5
                            * (1 - np.cos(beta)) ** (abs(m - m_prime) / 2) 
                            * (1 + np.cos(beta)) ** (abs(m + m_prime) / 2 ))
    
            for ll in range(l_min, l):
                wig_d[ll + 1] = (((2 * ll + 1) * (ll * (ll + 1) * np.cos(beta) - m * m_prime) * wig_d[ll] 
                                - (ll + 1) * (ll ** 2 - m ** 2) ** 0.5 * (ll ** 2 - m_prime ** 2) ** 0.5 
                                * wig_d[ll - 1]) / (ll * ((ll + 1) ** 2 - m ** 2) ** 0.5 
                                                    * ((ll + 1) ** 2 - m_prime ** 2) ** 0.5))
    
    else:
        wig_d[l] = complex(Rotation.d(l, m, m_prime, beta).doit())
      
    return wig_d[l].real

def wigner_D(l , m, m_prime, alpha, beta, gamma, wdsympy=False):
    """Computation of Wigner-D-functions for the rotation of a T-matrix
         
    Args:
        l (int):          Degree :math:`l` (1, ..., lmax)
        m (int):          Order :math:`m` (-min(l,mmax),...,min(l,mmax))
        m_prime (int):    Order :math:`m_prime` (-min(l,mmax),...,min(l,mmax))
        alpha (float):    First Euler angle in rad
        beta (float):     Second Euler angle in rad
        gamma (float):    Third Euler angle in rad
        wdsympy (bool):   If True, Wigner-d-functions come from the sympy toolbox
        
    Returns:
        single complex value of Wigner-D-function 
    """       
    # Doicu, Light Scattering by Systems of Particles, p. 271ff (B.33ff)   
    if m >= 0 and m_prime >= 0:
        delta_m_mprime = 1
    elif m >= 0 and m_prime < 0:
        delta_m_mprime = (-1) ** m_prime
    elif m < 0 and m_prime >= 0:
        delta_m_mprime = (-1) ** m
    elif m < 0 and m_prime < 0:
        delta_m_mprime = (-1) ** (m + m_prime)
        
    wig_D = ((-1) ** (m + m_prime) * np.exp(1j * m * alpha) * delta_m_mprime * wigner_d(l, m, m_prime, beta, wdsympy) 
            * np.exp(1j * m_prime * gamma))

    # Mishchenko, Scattering, Absorption and Emission of Light by small Particles, p.367 (B.38)
#    wig_D = np.exp(-1j * m * alpha) * wigner_d(l, m, m_prime, beta) * np.exp(-1j * m_prime * gamma)    
   
    return wig_D

    
def legendre_normalized(ct, st, lmax):
    r"""Return the normalized associated Legendre function :math:`P_l^m(\cos\theta)` and the angular functions
    :math:`\pi_l^m(\cos \theta)` and :math:`\tau_l^m(\cos \theta)`, as defined in
    `A. Doicu, T. Wriedt, and Y. A. Eremin: "Light Scattering by Systems of Particles", Springer-Verlag, 2006
    <https://doi.org/10.1007/978-3-540-33697-6>`_.
    Two arguments (ct and st) are passed such that the function is valid for general complex arguments, while the branch
    cuts are defined by the user already in the definition of st.

    Args:
        ct (ndarray): cosine of theta (or kz/k)
        st (ndarray): sine of theta (or kp/k), need to have same dimension as ct, and st**2+ct**2=1 is assumed
        lmax (int): maximal multipole order

    Returns:
        - ndarray plm[l, m, *ct.shape] contains :math:`P_l^m(\cos \theta)`. The entries of the list have same dimension as ct (and st)
        - ndarray pilm[l, m, *ct.shape] contains :math:`\pi_l^m(\cos \theta)`.
        - ndarray taulm[l, m, *ct.shape] contains :math:`\tau_l^m(\cos \theta)`.
    """
    if hasattr(ct, '__len__'):
        ct = np.array(ct, dtype=np.complex128)
    else:
        ct = np.array([ct], dtype=np.complex128)

    if hasattr(st, '__len__'):
        st = np.array(st, dtype=np.complex128)
    else:
        st = np.array([st], dtype=np.complex128)

    return legendre_normalized_numbed(ct, st, lmax)

def legendre_normalized_numbed(ct, st, lmax):
    plm = np.zeros((lmax+1, lmax+1, *ct.shape), dtype=np.complex128)
    pilm = np.zeros((lmax+1, lmax+1, *ct.shape), dtype=np.complex128)
    taulm = np.zeros((lmax+1, lmax+1, *ct.shape), dtype=np.complex128)
    pprimel0 = np.zeros((lmax+1, *ct.shape), dtype=np.complex128)

    plm[0,0] = np.sqrt(2)/2
    plm[1, 0] = np.sqrt(3/2) * ct
    pprimel0[1] = np.sqrt(3) * plm[0, 0]
    taulm[0, 0] = -st * pprimel0[0]
    taulm[1, 0] = -st * pprimel0[1]

    for l in range(1, lmax):
        plm[l + 1, 0] = (1 / (l + 1) * np.sqrt((2 * l + 1) * (2 * l + 3)) * ct * plm[l, 0] -
                         l / (l + 1) * np.sqrt((2 * l + 3) / (2 * l - 1)) * plm[l-1, 0])
        pprimel0[l + 1] = ((l + 1) * np.sqrt((2 * (l + 1) + 1) / (2 * (l + 1) - 1)) * plm[l, 0] +
                           np.sqrt((2 * (l + 1) + 1) / (2 * (l + 1) - 1)) * ct * pprimel0[l])
        taulm[l + 1, 0] = -st * pprimel0[l + 1]

    for m in range(1, lmax + 1):
        prefactor = prefactor_expansion(m)
        plm[m, m] = prefactor * st**m
        pilm[m, m] = prefactor * st**(m - 1)
        taulm[m, m] = m * ct * pilm[m, m]
        for l in range(m, lmax):
            plm[l + 1, m] = (np.sqrt((2 * l + 1) * (2 * l + 3) / ((l + 1 - m) * (l + 1 + m))) * ct * plm[l, m] -
                             np.sqrt((2 * l + 3) * (l - m) * (l + m) / ((2 * l - 1) * (l + 1 - m) * (l + 1 + m))) *
                             plm[l - 1, m])
            pilm[l + 1, m] = (np.sqrt((2 * l + 1) * (2 * l + 3) / (l + 1 - m) / (l + 1 + m)) * ct * pilm[l, m] -
                              np.sqrt((2 * l + 3) * (l - m) * (l + m) / (2 * l - 1) / (l + 1 - m) / (l + 1 + m)) *
                              pilm[l - 1, m])
            taulm[l + 1, m] = ((l + 1) * ct * pilm[l + 1, m] -
                               (l + 1 + m) * np.sqrt((2 * (l + 1) + 1) * (l + 1 - m) / (2 * (l + 1) - 1) / (l + 1 + m))
                               * pilm[l, m])

    return plm, pilm, taulm


def jitted_prefactor(m):
    r"""Returns the prefactor :math:`\sqrt(\frac{(2*m+1)}{2(2m)!} (2m-1)!!`
    without using factorials nor bignum numbers, which makes it jittable.

    Args:
        m (int64): Argument (non-negative)

    Returns:
        :math:`\sqrt(\frac{(2*m+1)!!}{2(2m)!!}`
    """
    res = 1.
    for t in range(2,2*m+2,2):
        res += res/t # equivalent to res *= (t+1)/t, but retains all significant digits
    return (res/2)**0.5


def prefactor_expansion(m):
    r"""Expansion of :math:`\sqrt(\frac{(2*m+1)}{2(2m)!} (2m-1)!!` in the limit
    for large :math:`m`.
    The expansion converges very rapidly, so that taking the first 14 terms is
    enough to get every term with double precision down to :math:`m > 10`.
    Further terms are available, but they involve bignum integers.
    Therefore, for terms :math:`m < 10`, it's easier to perform the calculation
    directly.

    Args:
        m (int64): Argument (non-negative)

    Returns:
        :math:`\sqrt(\frac{(2*m+1)!!}{2(2m)!!}`
    """
    if m <= 10:
        return jitted_prefactor(m)
    x = (1/m)**(1/4) / 2
    res = 1/2/x \
        + 3/2*x**3 \
        - 23/4*x**7 \
        + 105/4*x**11 \
        - 1317/16*x**15 \
        + 1053/16*x**19 \
        - 132995/32*x**23 \
        + 3300753/32*x**27 \
        + 24189523/256*x**31 \
        - 7404427407/256*x**35 \
        - 45203760489/512*x**39 \
        + 8818244857071/512*x**43 \
        + 99932439090703/2048*x**47 \
        - 29944926937328991/2048*x**51 \
        # - 173768350561954907/4096*x**55 \
        # + 70443346589090375073/4096*x**59 \
        # + 3299174696912539072131/65536*x**63 \
        # - 1750098951789039471641607/65536*x**67 \
        # - 10306911268683450183973389/131072*x**71 \
        # + 6934593111025074608358220131/131072*x**75 \
        # + 82009190686072900341958975797/524288*x**79 \
        # - 68275536676414175755318233771549/524288*x**83 \
        # - 404866365213072360301912455070149/1048576*x**87 \
        # + 408755654050262791361092019679647223/1048576*x**91 \
        # + 9715746540636352989024348199533962007/8388608*x**95 \
        # - 11697824487409665503661150266687972482035/8388608*x**99
    return res / (np.pi)**(1/4)
