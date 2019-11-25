import sqlite3
import numpy as np
import pandas as pd
import desc.imsim


__all__ = ['OpSimDb']


class OpSimDb:
    def __init__(self, opsim_db_file='/global/projecta/projectdirs/lsst/groups/SSim/DC2/minion_1016_desc_dithered_v4.db'):
        self.conn = sqlite3.connect(opsim_db_file)
        self.columns = '''obsHistID filter expMJD airmass vSkyBright altitude
                          azimuth dist2Moon rawSeeing fiveSigmaDepth
                          descDitheredRA descDitheredDec
                          descDitheredRotTelPos'''.split()

    def __call__(self, visit):
        query = f'''select {", ".join(self.columns)} from Summary where
                    obsHistID={visit} limit 1'''
        row = pd.read_sql(query, self.conn).iloc[0]
        row['band'] = row['filter']
        row['FWHMgeom'] = desc.imsim.FWHMgeom(row.rawSeeing, row.band,
                                              np.degrees(row.altitude))
        row['FWHMeff'] = desc.imsim.FWHMeff(row.rawSeeing, row.band,
                                            np.degrees(row.altitude))
        return row

