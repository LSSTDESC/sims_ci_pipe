"""
Code to extract point sources from a SourceCatalog.
"""
__all__ = ['get_band', 'get_point_sources']


def get_band(butler, dsref):
    dims = butler.registry.expandDataId(dsref.dataId)
    return dims.records['band'].name


def get_point_sources(src, flux_type='base_PsfFlux', min_snr=None, flags=()):
    ext = src.get('base_ClassificationExtendedness_value')
    model_flag = src.get(f'{flux_type}_flag')
    model_flux = src.get(f'{flux_type}_instFlux')
    num_children = src.get('deblend_nChild')
    snr = model_flux/src.get(f'{flux_type}_instFluxErr')
    condition = ((ext == 0) &
                 (model_flag == False) &
                 (model_flux > 0) &
                 (num_children == 0))
    for flag in flags:
        values = src.get(flag)
        condition &= (values == True)
    if min_snr is not None:
        condition &= (snr >= min_snr)
    return src.subset(condition).copy(deep=True)
