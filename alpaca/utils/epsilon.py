import numpy as np

class eps:

    #GOLD
    def epsAu(
            lambda_nm: float
    ) -> float:
        """
        Computes epsilon (refracive index squared) for gold

        Parameters
        ----------
        lambda_nm : float
            wavelength in nanometer.

        Returns
        -------
        epsAu: float
            epsilon (refractive index squared) for gold.
        """

        #gold-specif parameters
        eps_infty = 1.54
        lambda_nmbda_p = 177.5
        mu_p = 14500.0
        A1=1.27
        lambda_nmbda1=470.0
        mu_p1=1900.0
        A2=1.1
        lambda_nmbda2=325.0
        mu_p2=1060.0
        phi=-np.pi/4
        epsAu = eps_infty * (1 - 1 / (lambda_nmbda_p**2 *( (1/lambda_nm)**2 + 1j/(mu_p*lambda_nm)))) \
            + A1 / lambda_nmbda1 *(np.exp(1j*phi)/(1/lambda_nmbda1-1/lambda_nm-1j/mu_p1)+np.exp(-1j*phi)/(1/lambda_nmbda1+1/lambda_nm+1j/mu_p1)) \
            + A2 / lambda_nmbda2 *(np.exp(1j*phi)/(1/lambda_nmbda2-1/lambda_nm-1j/mu_p2)+np.exp(-1j*phi)/(1/lambda_nmbda2+1/lambda_nm+1j/mu_p2))
        return epsAu

    #SILVER
    def epsAg(
            lambda_nm: float
    ) -> float:
        """
        Computes epsilon (refracive index squared) for silver

        Parameters
        ----------
        lambda_nm : float
            wavelength in nanometer.

        Returns
        -------
        epsAg: float
            epsilon (refractive index squared) for silver.
        """
        #silver-specif parameters
        eps_infty = 4.0
        lambda_nmbda_p = 282.0
        mu_p = 17000.0
        epsAg = eps_infty *(1-1/(lambda_nmbda_p**2 *( (1/lambda_nm)**2 + 1j/(mu_p*lambda_nm))))
        return epsAg

    #POLYSTYRENE
    def nPSL(
            lambda_nm: float
    ) -> float:
        """
        Computes epsilon (refracive index squared) for polystryrene

        Parameters
        ----------
        lambda_nm : float
            wavelength in nanometer.

        Returns
        -------
        epsAg: float
            epsilon (refractive index squared) for polystryrene.
        """
        # N. Sultanova, S. Kasarova and I. Nikolov. Dispersion properties of optical polymers, Acta Physica Polonica A 116, 585-587 (2009)
        # (fit of the experimental data with the Sellmeier dispersion formula: Mikhail Polyanskiy)
        
        lambda_nm_PSL = lambda_nm/1000;     #nm > um
        n_PSL = np.sqrt((1.4435*lambda_nm_PSL**2)/(lambda_nm_PSL**2 - 0.020216)+1)
        return n_PSL