import numpy as np
import scipy
import vendored.smuthi.expansion as fldex

from alpaca.parameters import params_general
from utils.epsilon import eps
from vendored.smuthi.fields import multi_to_single_index
from vendored.smuthi.transformations import block_rotation_matrix_D_svwf, swe_to_pwe_conversion
from vendored.smuthi.layers import LayerSystem
from vendored.smuthi.vwf import plane_vector_wave_function


class PSFclass:
    def __init__(self,params: params_general):
        self.params = params
        
        # Assign parameters from params instance
        self.radius = params.radius
        self.d = params.d
        self.lam = params.lam
        self.epsilonM = params.epsilonM
        self.withoutParticle = params.withoutParticle
        self.focus_overwrite = params.focus_overwrite
        self.focus_overwrite_value = params.focus_overwrite_value
        self.angleDipole = params.angleDipole
        self.alpha = params.alpha
        self.beta = params.beta
        self.gamma = params.gamma
        self.dist_NP_glass = params.dist_NP_glass
        self.mode = params.mode
        self.n_water = params.n_water
        self.n_glass = params.n_glass
        self.l_max = params.l_max
        self.m_max = params.m_max
        self.N = params.N
        self.r = params.r
        self.M = params.M
        self.NA = params.NA
        self.z0 = params.z0
        self.num_px = params.num_px
        
        # Other calculations
        self.epsilonIn = eps.epsAu(params.lam)
        self.px_size = params.px_size * params.M

        self.radius_list = [params.radius]
        self.epsilon_list = [self.epsilonIn, params.epsilonM]

    def calc_PSF(self):
        #> main function to calculate the PSF

        # # STEP 1 - COEFFS+SWE
        k, swe_scatt_manual, swe_initial_manual = self.calc_coeffs_and_fill_swe()

        # #STEP 2 - ROTATE SWE
        swe_scatt_manual,swe_initial_manual     = self.rotate_SWEs(swe_scatt_manual,swe_initial_manual)

        # #STEP 3 - SPA
        spa_vectors, spa_fields                 = self.do_SPA_unitVec(swe_scatt_manual+swe_initial_manual,k)

        # #STEP 4 - FOCUS
        integral_output                         = self.do_focus_integral_fast_unitVec(spa_vectors,spa_fields)

        return integral_output

    def Suscep_helpers(self,n_list,x,s):
        #> helper function to calculate the required elements of the susceptibilities Gamma/Delta

        #Susceptibilities
        psi_x = x * scipy.special.spherical_jn(n_list, x)
        psi_sx = s*x * scipy.special.spherical_jn(n_list, s*x)
        Dpsi_x = x * scipy.special.spherical_jn(n_list-1, x) - n_list * psi_x/x
        Dpsi_sx =  s*x * scipy.special.spherical_jn(n_list-1, s*x) - n_list * psi_sx/s/x
        xi_x = x * (scipy.special.spherical_jn(n_list, x) +1j*scipy.special.spherical_yn(n_list, x))
        Dxi_x = x * (scipy.special.spherical_jn(n_list-1, x) +1j*scipy.special.spherical_yn(n_list-1, x)) - n_list * xi_x/x
        xi_sx = s*x * (scipy.special.spherical_jn(n_list, s*x) +1j*scipy.special.spherical_yn(n_list, s*x))
        Dxi_sx = s*x * (scipy.special.spherical_jn(n_list-1, s*x) +1j*scipy.special.spherical_yn(n_list-1, s*x)) - n_list * xi_sx/s/x
        return psi_x, psi_sx, Dpsi_x, Dpsi_sx, xi_x, xi_sx, Dxi_x, Dxi_sx

    def Suscep_coreShell(self, a_list, epsilon_list):
        #> calculate the susceptibilities Gamma/Delta

        n_list = np.arange(1,self.l_max+1)

        #check is radius_list has the same length as the epsilon_list
        if len(a_list) != (len(epsilon_list)-1):
            print('wrong dimensions a_list/epsilon_list !')

        Gamma = 0
        Delta = 0
        Gamma_list = []
        Delta_list = []
        Gamma_list.append(Gamma)
        Delta_list.append(Delta)

        for i in range(1,1+len(a_list)):
            a_i = a_list[i-1]      #a1 is first boundary (between core and shell) > a2 is boundary shell-water
            eps_im1 = epsilon_list[i-1]
            eps_i = epsilon_list[i]
            x_i = 2 * np.pi * np.sqrt(eps_i) * a_i / self.lam     # x=kM*a  #eq. H.45
            s_i = np.sqrt(eps_im1)/np.sqrt(eps_i)        # #eq. H.45

            #for boundary i:
            psi_x, psi_sx, Dpsi_x, Dpsi_sx, xi_x, xi_sx, Dxi_x, Dxi_sx = self.Suscep_helpers(n_list,x_i,s_i)
            PP1_g = (psi_sx + Gamma * xi_sx) * Dpsi_x        #p_sx_dp_x
            PP2_g = psi_x * (Dpsi_sx + Gamma * Dxi_sx)       #p_x_dp_sx
            PP3_g = (psi_sx + Gamma * xi_sx) * Dxi_x        #p_sx_dchi_x
            PP4_g = xi_x * (Dpsi_sx + Gamma * Dxi_sx)       #chi_x_dp_sx

            PP1_d = (psi_sx + Delta * xi_sx) * Dpsi_x        #p_sx_dp_x
            PP2_d = psi_x * (Dpsi_sx + Delta * Dxi_sx)       #p_x_dp_sx
            PP3_d = (psi_sx + Delta * xi_sx) * Dxi_x        #p_sx_dchi_x
            PP4_d = xi_x * (Dpsi_sx + Delta * Dxi_sx)       #chi_x_dp_sx

            Gamma = (s_i*PP2_g - PP1_g) / (PP3_g - s_i*PP4_g)
            Delta = (PP2_d - s_i*PP1_d) / (s_i*PP3_d - PP4_d)

            Gamma_list.append(Gamma)
            Delta_list.append(Delta)

        return Gamma, Delta

    def calc_coefficients(self):
        #> calculate the Mie coefficients

        n_list = np.arange(1,self.l_max+1)

        x = 2 * np.pi * np.sqrt(self.epsilonM) * self.radius / self.lam      # x=kM*a  #eq. H.45
        s = np.sqrt(self.epsilonIn)/np.sqrt(self.epsilonM)       # #eq. H.45
        xp = x/self.radius*(self.radius+self.d)                             # xp=k_M (a+d) for dipole position

        spherj = scipy.special.spherical_jn(n_list, xp)
        sphery = scipy.special.spherical_yn(n_list, xp)
        hn1 = spherj + 1j * sphery

        #perpendicular (fn0 + relatives)
        nCoeffs = np.sqrt((3/2)*n_list*(n_list+1)*(2*n_list+1))
        bn0 = nCoeffs * hn1/xp    #not used (internal field)
        fn0 = nCoeffs * spherj/xp

        #parallel (en1/fn1 + relatives)
        nCoeffs = np.sqrt((3/8)*(2*n_list+1))
        en1 = 1j * nCoeffs * spherj
        Z2_j =  scipy.special.spherical_jn(n_list-1, xp) - n_list * spherj / xp
        Z2_h =  scipy.special.spherical_jn(n_list-1, xp) + 1j*scipy.special.spherical_yn(n_list-1, xp) - n_list * hn1 / xp
        fn1 = -1 * nCoeffs * Z2_j
        bn1 = -1 * nCoeffs * Z2_h
        an1 = 1j * nCoeffs * hn1

        #Susceptibilities
        if self.mode=='solid':
            Gamma, Delta = self.Suscep_coreShell([self.radius],[self.epsilonIn,self.epsilonM])
        elif self.mode=='core-shell':
            Gamma, Delta = self.Suscep_coreShell(self.radius_list,self.epsilon_list)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'solid' or 'core-shell'.")

        cn1 = Gamma * an1
        dn1 = Delta * bn1
        dn0 = Delta * bn0

        coeffs = dict()
        coeffs['an1'] = an1     #
        coeffs['bn1'] = bn1     #these 3 are not necessary to save! (except for absorption cross section calculation)
        coeffs['bn0'] = bn0     #

        coeffs['cn1'] = cn1
        coeffs['dn0'] = dn0
        coeffs['dn1'] = dn1
        coeffs['en1'] = en1
        coeffs['fn0'] = fn0
        coeffs['fn1'] = fn1
        coeffs['Gamma'] = Gamma
        coeffs['Delta'] = Delta
        return coeffs

    def calc_coeffs_and_fill_swe(self):
        #> fill the spherical wave expansion (SWE) with the previously calculated Mie coefficients
        Mie_coeffs = self.calc_coefficients()
        cn1 = Mie_coeffs['cn1']
        dn1 = Mie_coeffs['dn1']
        en1 = Mie_coeffs['en1']
        fn1 = Mie_coeffs['fn1']
        dn0 = Mie_coeffs['dn0']
        fn0 = Mie_coeffs['fn0']

        pz_factor = round(np.cos(self.angleDipole),6)   #perp
        px_factor = round(np.sin(self.angleDipole),6)   #para  (round because cos(pi/2)=1e-17 not 0)

        k = 2*np.pi / self.lam * self.n_glass   #wave vector in glass

        swe_scat_man_perp = fldex.SphericalWaveExpansion(k=2*np.pi/self.lam*self.n_water,    #water!!
                                        l_max=self.l_max,
                                        m_max=self.m_max,
                                        kind='outgoing',
                                        reference_point=[0,0,self.radius+self.dist_NP_glass],            #because NP is at 50
                                        lower_z = 0)
        swe_scat_man_para = fldex.SphericalWaveExpansion(k=2*np.pi/self.lam*self.n_water,    #water!!
                                        l_max=self.l_max,
                                        m_max=self.m_max,
                                        kind='outgoing',
                                        reference_point=[0,0,self.radius+self.dist_NP_glass],            #because NP is at 50
                                        lower_z = 0)
        swe_init_man_perp = fldex.SphericalWaveExpansion(k=2*np.pi/self.lam*self.n_water,    #water!!
                                        l_max=self.l_max,
                                        m_max=self.m_max,
                                        kind='outgoing',
                                        reference_point=[0,0,self.radius+self.dist_NP_glass],            #because NP is at 50
                                        lower_z = 0)
        swe_init_man_para = fldex.SphericalWaveExpansion(k=2*np.pi/self.lam*self.n_water,    #water!!
                                        l_max=self.l_max,
                                        m_max=self.m_max,
                                        kind='outgoing',
                                        reference_point=[0,0,self.radius+self.dist_NP_glass],            #because NP is at 50
                                        lower_z = 0)

        # REFILL COEFFICIENTS WITH MIE COEFFS
        for m in np.arange(1,self.l_max+1):
            ## Perpendicular:
            n = multi_to_single_index(tau = 1,       #0>M, 1>N
                                        l = m,         #'n'
                                        m = 0,
                                        l_max=self.l_max,
                                        m_max=self.m_max)
            swe_scat_man_perp.coefficients[n] = pz_factor * dn0[m-1]*1j     #WHATCH OUT for the 1j
            swe_init_man_perp.coefficients[n] = pz_factor * fn0[m-1]*1j


            ## Parallel:
            n = multi_to_single_index(tau = 1,       #0>M, 1>N
                                        l = m,         #'n'
                                        m = 1,
                                        l_max=self.l_max,
                                        m_max=self.m_max)
            swe_scat_man_para.coefficients[n] = px_factor * dn1[m-1]*-1j #MINUS DUE TO CONVERSION FACTOR M=1
            swe_init_man_para.coefficients[n] = px_factor * fn1[m-1]*-1j


            n = multi_to_single_index(tau = 1,       #0>M, 1>N
                                        l = m,         #'n'
                                        m = -1,
                                        l_max=self.l_max,
                                        m_max=self.m_max)
            swe_scat_man_para.coefficients[n] = px_factor * dn1[m-1]*-1j ##MINUS DUE TO F=-F
            swe_init_man_para.coefficients[n] = px_factor * fn1[m-1]*-1j


            n = multi_to_single_index(tau = 0,       #0>M, 1>N
                                        l = m,         #'n'
                                        m = 1,
                                        l_max=self.l_max,
                                        m_max=self.m_max)
            swe_scat_man_para.coefficients[n] = px_factor * cn1[m-1]*-1j #MINUS DUE TO CONVERSION FACTOR M=1
            swe_init_man_para.coefficients[n] = px_factor * en1[m-1]*-1j


            n = multi_to_single_index(tau = 0,       #0>M, 1>N
                                        l = m,         #'n'
                                        m = -1,
                                        l_max=self.l_max,
                                        m_max=self.m_max)
            swe_scat_man_para.coefficients[n] = px_factor * cn1[m-1]*1j
            swe_init_man_para.coefficients[n] = px_factor * en1[m-1]*1j  ##the j is just to match Lumerical (right?)
                                                            #and constant phase does not change PSF

        swe_scatt_manual = swe_scat_man_perp + swe_scat_man_para
        swe_initial_manual = swe_init_man_perp + swe_init_man_para

        return k, swe_scatt_manual, swe_initial_manual

    def rotate_SWEs(self, swe_scatt_manual, swe_initial_manual):
        #> rotate the emitter around the NP (by rotating the SWE)
        D = block_rotation_matrix_D_svwf(self.l_max,self.m_max,self.alpha,self.beta,self.gamma,False)

        swe_scatt_manual.coefficients   = np.matmul(D,swe_scatt_manual.coefficients)
        swe_initial_manual.coefficients = np.matmul(D,swe_initial_manual.coefficients)

        return swe_scatt_manual,swe_initial_manual

    def do_SPA_unitVec(self, swe_to_use, k):
        #> perform stationary phase approximation (=SWE to PWE)

        #change focus if desired
        if self.focus_overwrite == True:
            focus = self.focus_overwrite_value
        else:
            focus = self.radius + self.dist_NP_glass  #nm

        # xyz = dome around around (0,0,+focus)
        polar_wanted = np.linspace(0.501*np.pi,np.pi,self.N)      #pi/2>pi (pi/2 gives error in swe2pwe!)
                                                        # #0.500000016pi is enough to have dome in glass
        x = self.r * np.sin(polar_wanted)
        z = self.r * np.cos(polar_wanted)                   #negative (50 > -(1e9-50))
        z = z + focus

        # create coordinates for system focused arount (0,0,0)
        thetas_wanted = np.arctan2(x,z)               #~pi/2 > pi (0.49999pi>1pi)
        kappa_wanted = k*np.sin(thetas_wanted)        #k>0
        alphas_wanted = np.linspace(0,2*np.pi*(1-1/self.N),self.N)  #0>~2pi


        ## MAKE PWE DOWNs FOR BOTH FIELDS (scattering + initial)
        layer_system = LayerSystem(thicknesses=[0, 0],
                                                refractive_indices=[self.n_glass, self.n_water])

        _ , pwe_down = swe_to_pwe_conversion(swe_to_use,
                                                    k_parallel=kappa_wanted,
                                                    azimuthal_angles=alphas_wanted,
                                                    layer_system=layer_system, layer_number=0,
                                                    layer_system_mediated=True)

        #GET KAPPAS/ALPHAS grids
        kappas_grid = pwe_down.k_parallel_grid()       # (NxN)
        alphas_grid = pwe_down.azimuthal_angle_grid()  # (NxN)

        #DO SPA
        #get alpha, kappa, kz and thetas
        alphas_vector = alphas_grid.flatten()           #N^2x1       0>2pi (N inclines)
        kappas_vector = kappas_grid.flatten()           #N^2x1       k>0   (in N steps of N points)
        kz_vector = -np.sqrt(k**2 - kappas_vector**2)   #N^2x1       0>-k  (in N steps of N points)
        thetas_vector = np.arccos(kz_vector/k)          #N^2x1       pi/2>pi  (in N steps of N points)

        #layer-reflected scattered field
        xtilde = np.repeat(x*np.cos(alphas_wanted),self.N)
        ytilde = np.repeat(x*np.sin(alphas_wanted),self.N)
        ztilde = np.abs(np.repeat(z,self.N))                #positive (but not used)
        rtilde = np.sqrt(xtilde**2 + ytilde**2 + ztilde**2)

        prefac = -1j * np.exp(1j * k * rtilde) / (k * rtilde)

        Exs_spa = np.zeros(len(thetas_vector),dtype=complex)
        Eys_spa = np.zeros(len(thetas_vector),dtype=complex)
        Ezs_spa = np.zeros(len(thetas_vector),dtype=complex)
        Hxs_spa = np.zeros(len(thetas_vector),dtype=complex)
        Hys_spa = np.zeros(len(thetas_vector),dtype=complex)
        Hzs_spa = np.zeros(len(thetas_vector),dtype=complex)

        #for both polarizations:
        for pol in range(2):

            pol_H = 1-pol
            K2H_fac = (2*pol-1)*k
            coeff = pwe_down.coefficients[pol,:,:].flatten()
            scalar_fac = prefac * (2*np.pi*kz_vector*k) * coeff

            if pol==0:
                E_scalar_fac_pol0 = scalar_fac
                H_scalar_fac_pol0 = scalar_fac * K2H_fac    #K2H_fac=-1*k
            elif pol==1:
                E_scalar_fac_pol1 = scalar_fac
                H_scalar_fac_pol1 = scalar_fac * K2H_fac    #K2H_fac=1*k
            else:
                print('WRONG POL in do_SPA_unitVec!!')

            pvwf_direction = plane_vector_wave_function(0, 0, 0, kappas_vector, alphas_vector, kz_vector, pol)
            pvwf_direction_H = plane_vector_wave_function(0, 0, 0, kappas_vector, alphas_vector, kz_vector, pol_H)

            Exs_spa += pvwf_direction[0] * scalar_fac
            Eys_spa += pvwf_direction[1] * scalar_fac
            Ezs_spa += pvwf_direction[2] * scalar_fac
            Hxs_spa += pvwf_direction_H[0] * scalar_fac * K2H_fac
            Hys_spa += pvwf_direction_H[1] * scalar_fac * K2H_fac
            Hzs_spa += pvwf_direction_H[2] * scalar_fac * K2H_fac

        vectors = dict()
        vectors['alphas_vector'] = alphas_vector
        vectors['kappas_vector'] = kappas_vector
        vectors['kz_vector'] = kz_vector
        vectors['thetas_vector'] = thetas_vector

        fields = dict()
        fields['E_scalar_fac_pol0'] = np.flipud(E_scalar_fac_pol0.reshape((self.N, self.N)))
        fields['E_scalar_fac_pol1'] = np.flipud(E_scalar_fac_pol1.reshape((self.N, self.N)))
        fields['H_scalar_fac_pol0'] = np.flipud(H_scalar_fac_pol0.reshape((self.N, self.N)))
        fields['H_scalar_fac_pol1'] = np.flipud(H_scalar_fac_pol1.reshape((self.N, self.N)))

        fields['Exs_spa'] = Exs_spa       #Exs_spa saved for easy plotting in Matlab
        fields['Eys_spa'] = Eys_spa
        fields['Ezs_spa'] = Ezs_spa
        fields['Hxs_spa'] = Hxs_spa
        fields['Hys_spa'] = Hys_spa
        fields['Hzs_spa'] = Hzs_spa

        return vectors, fields

    def do_focus_integral_fast_unitVec(self,vectors,fields):
        #calculate the focussing integral
        n1 = self.n_water   #water
        n2 = self.n_glass   #glass
        n3 = 1      #air

        k3 = 2*np.pi/self.lam * n3

        dtheta = (np.max(vectors["thetas_vector"])-np.min(vectors["thetas_vector"]))/self.N # ~ pi/2/N
        dalpha = (np.max(vectors["alphas_vector"])-np.min(vectors["alphas_vector"]))/self.N # = 2pi(1-1/N)/N

        thetas_reshaped = np.flipud(vectors['thetas_vector'].reshape((self.N,self.N)))   #NxN
        alphas_reshaped = np.flipud(vectors['alphas_vector'].reshape((self.N,self.N)))   #NxN

        #object coordinates
        theta1 = np.pi-thetas_reshaped #(??)                  #NxN,   GLASS (so means eta_2) = measured negative z-axis to ray
        theta2 = np.arcsin( n2/(n3*self.M) * np.sin(theta1))  #NxN    AIR (so means eta_prime) = measured negative z-axis to ray
        phii = alphas_reshaped
        theta1 = theta1[:,:,np.newaxis]
        theta2 = theta2[:,:,np.newaxis]
        phii = phii[:,:,np.newaxis]

        #filter NA
        theta_max = np.arcsin(self.NA/n2)          # maximum angle due to NA (eq 4.11) > max of theta_glass!
        filter_NA = theta1 <= theta_max    # 1 for angles <= theta_max

        #image coordinates
        x_image = np.linspace(-(self.num_px-1)/2*self.px_size,(self.num_px-1)/2*self.px_size,self.num_px)     #num_px,num_px. (increases left-to-right)
        x_image, y_image = np.meshgrid(x_image, x_image)
        z_image = self.z0
        rho_image = np.sqrt(x_image**2 + y_image**2)
        phi_image = np.arctan2(y_image,x_image)

        rho_image_flat = rho_image.flatten()          #concatenate per row, starting first/top row
        phi_image_flat = phi_image.flatten()
        rho_image_flat = rho_image_flat[np.newaxis,np.newaxis, :]
        phi_image_flat = phi_image_flat[np.newaxis,np.newaxis, :]

        #initialize arrays
        Ex_image_flat = np.zeros(self.num_px*self.num_px,dtype=complex)
        Ey_image_flat = np.zeros(self.num_px*self.num_px,dtype=complex)
        Ez_image_flat = np.zeros(self.num_px*self.num_px,dtype=complex)
        Hx_image_flat = np.zeros(self.num_px*self.num_px,dtype=complex)
        Hy_image_flat = np.zeros(self.num_px*self.num_px,dtype=complex)
        Hz_image_flat = np.zeros(self.num_px*self.num_px,dtype=complex)

        #perform integral
        block_size = 500
        num_blocks = int(np.ceil(self.num_px**2 / block_size))

        for b in range(num_blocks):
            #index the correct pixels per block
            if ((b+1)*block_size) < (self.num_px**2):
                ind = np.arange(block_size) + b*block_size
            else:
                ind = np.setdiff1d(np.arange(self.num_px**2),np.arange(b*block_size))

            A = 1       #A = -1i*dip.k(3)*f(2)*exp(-1i*dip.k(3)*f(2))/(2*pi) (???)
            temp = 1j *  k3 * (z_image * np.cos(theta2) + rho_image_flat[:,:,ind] * np.sin(theta2) * np.cos(phii-phi_image_flat[:,:,ind]))
            F = np.exp(temp)
            F = F * np.sin(theta2)
            F = F * np.sqrt( (n3 * np.cos(theta2)) / (n2 * np.cos(theta1)) )
            F = F * filter_NA
            F = F * dtheta * dalpha * (self.r)   #integral Jacobian [ADDED FEB 17, 2023]

            es_x = -np.sin(phii)
            es_y = np.cos(phii)
            es_z = 0

            ep_x = -np.cos(phii) * np.cos(theta2)
            ep_y = -np.sin(phii) * np.cos(theta2)
            ep_z = np.sin(theta2)

            Fx = (es_x * fields['E_scalar_fac_pol0'][:,:,np.newaxis] + ep_x * fields['E_scalar_fac_pol1'][:,:,np.newaxis]) * F
            Fy = (es_y * fields['E_scalar_fac_pol0'][:,:,np.newaxis] + ep_y * fields['E_scalar_fac_pol1'][:,:,np.newaxis]) * F
            Fz = (es_z * fields['E_scalar_fac_pol0'][:,:,np.newaxis] + ep_z * fields['E_scalar_fac_pol1'][:,:,np.newaxis]) * F

            #removed minus in front of ep, because the (minus)k is already in the fields (because of K2H_fac used before)
            F2x = (ep_x * fields['H_scalar_fac_pol0'][:,:,np.newaxis] + es_x * fields['H_scalar_fac_pol1'][:,:,np.newaxis]) * F
            F2y = (ep_y * fields['H_scalar_fac_pol0'][:,:,np.newaxis] + es_y * fields['H_scalar_fac_pol1'][:,:,np.newaxis]) * F
            F2z = (ep_z * fields['H_scalar_fac_pol0'][:,:,np.newaxis] + es_z * fields['H_scalar_fac_pol1'][:,:,np.newaxis]) * F
            #ALL CALCULATION OF ES/EP AND THE UNIT VECTOR PROJECTIONS CAN BE DONE 'OUTSIDE' FORLOOP FOR SPEED

            Ex_image_flat[ind] = A * np.sum(Fx,axis=(0,1))
            Ey_image_flat[ind] = A * np.sum(Fy,axis=(0,1))
            Ez_image_flat[ind] = A * np.sum(Fz,axis=(0,1))
            Hx_image_flat[ind] = A * np.sum(F2x,axis=(0,1))
            Hy_image_flat[ind] = A * np.sum(F2y,axis=(0,1))
            Hz_image_flat[ind] = A * np.sum(F2z,axis=(0,1))

        Ex_image = Ex_image_flat.reshape(self.num_px,self.num_px)
        Ey_image = Ey_image_flat.reshape(self.num_px,self.num_px)
        Ez_image = Ez_image_flat.reshape(self.num_px,self.num_px)
        Hx_image = Hx_image_flat.reshape(self.num_px,self.num_px)
        Hy_image = Hy_image_flat.reshape(self.num_px,self.num_px)
        Hz_image = Hz_image_flat.reshape(self.num_px,self.num_px)

        #construct PSF
        E2_image = Ex_image*Ex_image.conjugate() + Ey_image*Ey_image.conjugate() + Ez_image*Ez_image.conjugate()
        H2_image = Hx_image*Hx_image.conjugate() + Hy_image*Hy_image.conjugate() + Hz_image*Hz_image.conjugate()
        Sz = Ex_image*Hy_image.conjugate() - Ey_image*Hx_image.conjugate()
        PSF = -Sz.real

        integral_output=dict()
        integral_output['theta_glass'] = theta1
        integral_output['theta_air'] = theta2
        integral_output['phi'] = phii
        integral_output['theta_max'] = theta_max
        integral_output['filter_NA'] = filter_NA
        integral_output['x_image'] = x_image
        integral_output['y_image'] = y_image
        integral_output['z_image'] = z_image
        integral_output['rho_image'] = rho_image
        integral_output['phi_image'] = phi_image
        integral_output['Ex_image'] = Ex_image
        integral_output['Ey_image'] = Ey_image
        integral_output['Ez_image'] = Ez_image
        integral_output['Hx_image'] = Hx_image
        integral_output['Hy_image'] = Hy_image
        integral_output['Hz_image'] = Hz_image
        integral_output['E2_image'] = E2_image
        integral_output['H2_image'] = H2_image
        integral_output['Sz'] = Sz
        integral_output['PSF'] = PSF

        return integral_output