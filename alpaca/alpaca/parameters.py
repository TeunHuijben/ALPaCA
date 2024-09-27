from utils.epsilon import eps

class params_general:
    def __init__(self, radius=100, d=5, lam=680, epsilonM=1.333**2, withoutParticle=False, 
                 focus_overwrite=False, focus_overwrite_value=0, angleDipole=0, alpha=0, 
                 beta=0, gamma=0, dist_NP_glass=6, mode='solid', n_water=1.333, 
                 n_glass=1.52, l_max=10, m_max=3, N=100, r=1e9, M=100, NA=1.49, 
                 z0=0, num_px=10, px_size=65):

        #emitter
        self.lam                = lam           #wavelength
        self.alpha              = alpha         #azimuthal position of the dipole
        self.beta               = beta          #...
        self.gamma              = gamma
        self.angleDipole        = angleDipole
        self.dist_NP_glass      = dist_NP_glass

        #nanoparticle
        self.radius             = radius
        self.d                  = d
        self.withoutParticle    = withoutParticle
        self.mode               = mode
        self.epsilonIn          = eps.epsAu(self.lam)

        #microscope
        self.N                  = N             #number of farfield angles (NxN)
        self.r                  = r             #radius of farfield hemisphere
        self.M                  = M             #magnification
        self.NA                 = NA            #numerical aperture
        self.z0                 = z0            #defocus of the microscope (nm)
        self.num_px             = num_px        #number of pixels in the image
        self.px_size            = px_size       #size of a pixel in the image (nm)

        #system
        self.epsilonM = epsilonM
        self.focus_overwrite = focus_overwrite
        self.focus_overwrite_value = focus_overwrite_value
        self.n_water = n_water
        self.n_glass = n_glass
        self.l_max = l_max
        self.m_max = m_max

# Define a default parameter set for gold
class params_gold(params_general):
    def __init__(self):
        super().__init__(radius=100, lam=680)
        self.epsilonIn          = eps.epsAu(self.lam)

# Define a default parameter set for polystryrene
class params_polystyrene(params_general):
    def __init__(self):
        super().__init__(radius=500, lam=680)
        self.epsilonIn          = (eps.nPSL(self.lam))**2

