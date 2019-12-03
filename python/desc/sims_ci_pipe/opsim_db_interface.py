import sqlite3
import numpy as np
import pandas as pd
try:
    import desc.imsim as desc_imsim
except ImportError as eobj:
    print(eobj)
    desc_imsim = None


__all__ = ['OpSimDb']


def airmass(altitude):
    """
    Function to compute the airmass from altitude using equation 3
    of Krisciunas and Schaefer 1991.

    Parameters
    ----------
    altitude: float
        Altitude of pointing direction in degrees.

    Returns
    -------
    float: the airmass in units of sea-level airmass at the zenith.
    """
    altRad = np.radians(altitude)
    return 1.0/np.sqrt(1.0 - 0.96*(np.sin(0.5*np.pi - altRad))**2)


def FWHMeff(rawSeeing, band, altitude):
    """
    Compute the effective FWHM for a single Gaussian describing the PSF.

    Parameters
    ----------
    rawSeeing: float
        The "ideal" seeing in arcsec at zenith and at 500 nm.
        reference: LSST Document-20160
    band: str
        The LSST ugrizy band.
    altitude: float
        The altitude in degrees of the pointing.

    Returns
    -------
    float: Effective FWHM in arcsec.
    """
    X = airmass(altitude)

    # Find the effective wavelength for the band.
    wl = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]

    # Compute the atmospheric contribution.
    FWHMatm = rawSeeing*(wl/500)**(-0.3)*X**(0.6)

    # The worst case instrument contribution (see LSE-30).
    FWHMsys = 0.4*X**(0.6)

    # From LSST Document-20160, p. 8.
    return 1.16*np.sqrt(FWHMsys**2 + 1.04*FWHMatm**2)


def FWHMgeom(rawSeeing, band, altitude):
    """
    FWHM of the "combined PSF".  This is FWHMtot from
    LSST Document-20160, p. 8.

    Parameters
    ----------
    rawSeeing: float
        The "ideal" seeing in arcsec at zenith and at 500 nm.
        reference: LSST Document-20160
    band: str
        The LSST ugrizy band.
    altitude: float
        The altitude in degrees of the pointing.

    Returns
    -------
    float: FWHM of the combined PSF in arcsec.
    """
    return 0.822*FWHMeff(rawSeeing, band, altitude) + 0.052


class OpSimDb:
    def __init__(self, opsim_db_file=None):
        if opsim_db_file is None:
            opsim_db_file = '/global/projecta/projectdirs/lsst/groups/SSim/DC2/minion_1016_desc_dithered_v4.db'
        self.conn = sqlite3.connect(opsim_db_file)
        self.columns = '''obsHistID filter expMJD airmass vSkyBright altitude
                          azimuth dist2Moon rawSeeing fiveSigmaDepth
                          descDitheredRA descDitheredDec
                          descDitheredRotTelPos'''.split()

    def __call__(self, visit):
        query = f'''select {", ".join(set(self.columns))} from Summary where
                    obsHistID={visit} limit 1'''
        row = self.query(query).iloc[0]
        row['band'] = row['filter']
        altitude = np.degrees(row.altitude)
        row['FWHMgeom'] = FWHMgeom(row.rawSeeing, row.band, altitude)
        row['FWHMeff'] = FWHMeff(row.rawSeeing, row.band, altitude)
        return row

    def query(self, query):
        return pd.read_sql(query, self.conn)
