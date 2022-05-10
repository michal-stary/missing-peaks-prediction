from matchms import Spectrum
import numpy as np
import os

def enhance_spectra(spectrums, new_peaks, l=50):
    assert len(spectrums) == len(new_peaks)
    for i, s in enumerate(spectrums):
        #print([s.peaks.mz, new_peaks[i][:l][0]])
        mz = np.concatenate([s.peaks.mz, new_peaks[i][:l].astype(float)])
        intensities = np.concatenate([s.peaks.intensities, np.repeat(0.1, l)])
        
        p = mz.argsort()
        mz= mz[p]
        intensities = intensities[p]
        
        yield Spectrum(mz=mz,
                       intensities=intensities,
                       metadata=s.metadata,
                       metadata_harmonization=False)
        
def predict_spectra(spectrums, new_peaks, l=50):
    assert len(spectrums) == len(new_peaks)
    for i, s in enumerate(spectrums):
        #print([s.peaks.mz, new_peaks[i][:l][0]])
        mz = np.concatenate([new_peaks[i][:l].astype(float)])
        intensities = np.concatenate([np.repeat(0.1, l)])
        
        p = mz.argsort()
        mz= mz[p]
        intensities = intensities[p]
        
        yield Spectrum(mz=mz,
                       intensities=intensities,
                       metadata=s.metadata,
                       metadata_harmonization=False)
        
def get_galaxy_filename(name, directory, prohibited=None):
    for filename in os.listdir(directory):
        if name in filename and (prohibited is None or all([x not in filename for x in prohibited])):
            return directory + "/" + filename