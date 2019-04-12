import warnings
import numpy as np
import lsst.daf.persistence as dp

def get_wcs(butler, dataId=None):
    if dataId is None:
        dataId = dict(visit=179279, raftName='R22', detectorName='S11')
    raw = butler.get('raw', dataId=dataId)
    return raw.getWcs()

def shuffled_mags(star_cat=None, mag_range=(16.3, 21)):
    if star_cat is None:
        star_cat = 'v169767-y_grid/star_cat_169767.txt'
    mags = []
    with open(star_cat, 'r') as fd:
        for line in fd:
            tokens = line.split()
            mag = float(tokens[4])
            if mag > mag_range[0] and mag < mag_range[1]:
                mags.append(mag)
    np.random.shuffle(mags)
    return np.array(mags)

template = "object {id} {ra:.15f} {dec:.15f} {mag:.8f} starSED/phoSimMLT/lte037-5.5-1.0a+0.4.BT-Settl.spec.gz 0 0 0 0 0 0 point none CCM 0.04056722 3.1\n"

x_pixels = np.linspace(150, 4050, 40)
y_pixels = np.linspace(100, 3900, 39)
num_stars = len(x_pixels)*len(y_pixels)
mags = shuffled_mags()[:num_stars]
mags.sort()

butler = dp.Butler('repo')
detectors = range(189)

visit = 169767
band = 'y'

outfile = 'v{visit}-{band}_grid/star_grid_{visit}.txt'.format(**locals())
missing_detectors = []
with open(outfile, 'w') as output:
    id = 0
    for detector in detectors:
        print("processing", detector)
        dataId = dict(visit=visit, detector=detector)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                wcs = get_wcs(butler, dataId=dataId)
            except Exception as eobj:
                print(eobj)
                missing_detectors.append(detector)
        for x_pix in x_pixels:
            for y_pix in y_pixels:
                ra, dec = [_.asDegrees() for _ in wcs.pixelToSky(x_pix, y_pix)]
                mag = mags[id % num_stars]
                output.write(template.format(**locals()))
                id += 1
print(missing_detectors)
