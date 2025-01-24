###########################################################
# directories

Dir = '/home/heh15/research/Antennae/'
scriptDir = Dir + 'scripts/'
regionDir = Dir + 'regions/'
imageDir_band3 = Dir + '2018/cont_100GHz/image/'
imageDir_band6 = Dir + '2018/cont_200GHz/image/'
imageDir_band3_2016 = Dir + '2016/Band3/'

sourcefile_band3 = regionDir + 'source_band3.crtf'
imagefile_band3 = imageDir_band3 + 'ngc4038_band3_cont_12m_7m.pbcor'

sourcefile_band6 = regionDir + 'source_band6.crtf'
imagefile_band6 = imageDir_band6 + 'ngc4038_band6_cont_12m_7m.pbcor' 

sourcefile_band3_2016 = regionDir + 'starcluster_band3_2016.crtf'
imagefile_band3_2016 = imageDir_band3_2016 + \
                       'ngc40389overlap_band3_uvrange_robust_2_smooth.pbcor'

###########################################################
# basic settings

ratio = 2.0

bands = ['band 3', 'band 6', 'band3_2016', 'band3_2016_p5', 'band7_2016', 'band7_2016_p5',
        'band3_2016_p5_p9', 'band3_p9_multiple', 'band3_p9_1b', 'HST_Pbeta', 'HST_Pbeta2',
        'band3_2016_2_p201', 'band7_2016_2_p201', 'band7_2016_p5_band3Ap', 'HST_I', 
        'band7_2016_band3Ap']
continuum = dict.fromkeys(bands)

# band 3 image parameters for 2018 data
continuum['band 3'] = {}
continuum['band 3']['imagename'] = imagefile_band3
continuum['band 3']['region'] = sourcefile_band3
continuum['band 3']['outfile'] = regionDir + 'source_band3_imfit.crtf'
continuum['band 3']['pbimage'] = imageDir_band3 + 'ngc4038_band3_cont_12m_7m.pb'
continuum['band 3']['rms'] = None
 
# band 6 image parameters for 2018 data. 
continuum['band 6'] = {}
continuum['band 6']['imagename'] = imagefile_band6
continuum['band 6']['region'] = sourcefile_band6
continuum['band 6']['outfile'] = regionDir + 'source_band3_imfit.crtf'
continuum['band 6']['pbimage'] = imageDir_band6 + 'ngc4038_band6_cont_12m_7m.pb'
continuum['band 6']['rms'] = None


# parameters for 2016 data in band 3. 
continuum['band3_2016'] = {}
continuum['band3_2016']['imagename'] = imagefile_band3_2016
continuum['band3_2016']['region'] = sourcefile_band3_2016
continuum['band3_2016']['outfile'] = regionDir + 'source_band3_2016_imfit.crtf'
continuum['band3_2016']['pbimage'] = imageDir_band3_2016 + 'ngc40389overlap_band3_uvrange_robust_2.pb'

# parameters for 2016 data in band 3 but with higher resolution
sys.path.append('configs')
import copy

from imfit_params_band3_2016_p5 import params
continuum['band3_2016_p5'] = copy.deepcopy(params)
continuum['band3_2016_p5']['pbcor'] = np.array([0.86, 0.87, 0.89, 0.58, 0.55, 0.99, 0.99, 0.96, 0.76]) 
continuum['band3_2016_p5']['rms'] = 1.5e-5

# parameters for 2016 data in band 7. 
from imfit_params_band7_2016 import params
continuum['band7_2016'] = copy.deepcopy(params)
continuum['band7_2016']['rms'] = None

# parameters for 2016 data in band 7 with robust=0.5
from imfit_params_band7_2016_p5 import params
continuum['band7_2016_p5'] = copy.deepcopy(params)
continuum['band7_2016_p5']['rms'] = 4.1e-5 
continuum['band7_2016_p5']['pbcor'] = np.array([0.93])

# parameters for 2016 data in band7 but using band 3 apertures
from imfit_params_band7_2016_p5_band3Ap import params 
continuum['band7_2016_p5_band3Ap'] = copy.deepcopy(params)
continuum['band7_2016_p5_band3Ap']['rms'] = 4.1e-5

# parameters for 2016 data in band7 but using band 3 apertures
from imfit_params_band7_2016_band3Ap import params
continuum['band7_2016_band3Ap'] = copy.deepcopy(params)
continuum['band7_2016_band3Ap']['rms'] = 4.1e-5


# parameters for 2016 data in band 3, robust 0.5, smoothed to 0.9 arcsec
from imfit_params_band3_2016_p5_p9 import params
continuum['band3_2016_p5_p9'] = copy.deepcopy(params)
continuum['band3_2016_p5_p9']['rms'] = None

# Fit the source that broken into multiple sources in Band3 2016 image 
# with resolution of 0.9 arcsec. 

from imfit_params_multiple import params
continuum['band3_p9_multiple'] = copy.deepcopy(params)
continuum['band3_p9_multiple']['rms'] = None

# parameters to fit the source 1b with double gaussian
from imfit_params_1b import params
continuum['band3_p9_1b'] = copy.deepcopy(params)
continuum['band3_p9_1b']['rms'] = None

# parameters to fit the source with image smoothed to the optical resolution
from imfit_params_band3_2016_p201arcsec import params
continuum['band3_2016_2_p201'] = params

from imfit_params_band7_2016_p201arcsec import params
continuum['band7_2016_2_p201'] = params 

# parameters to fit for Pbeta images
from imfit_params_HST_Pbeta import params
continuum['HST_Pbeta'] = copy.deepcopy(params)
continuum['HST_Pbeta']['rms'] = None 

# parameters to fit for Pbeta images to compare with GMC continuum
from imfit_params_HST_Pbeta2 import params
continuum['HST_Pbeta2'] = copy.deepcopy(params)
continuum['HST_Pbeta2']['rms'] = None

# parameters to fit for Pbeta images after continuum subtraction
from imfit_params_HST_Pbeta import params
continuum['HST_Pbeta_contsub'] = copy.deepcopy(params)
continuum['HST_Pbeta_contsub']['imagename'] = '../HST/Antennae_HST_Pbeta_contsub_header.fits' 
continuum['HST_Pbeta_contsub']['region'] = '../regions/source_HST_Pbeta_init_contsub.crtf'
continuum['HST_Pbeta_contsub']['outfile'] = '../regions/source_HST_Pbeta_imfit_contsub.crtf'
bands.append('HST_Pbeta_contsub')

# parameters to fit for I band images to compare with band 3 YMC continuum
from imfit_params_HST_I import params
continuum['HST_I'] = copy.deepcopy(params)
continuum['HST_I'] = copy.deepcopy(params)

###########################################################
# function

def beam_get(imagename, region_init, ratio=2.0, **kwarg):
    ''' 
    parameters
    imagename: The path to the CASA image file
    regions: The path to the CASA region file or the CASA 
    region string. 
    ratio: The ratio to be multiplied by the fitted ellipse to 
    get the final elliptical aperture shape.
    **kwarg: other parameters that goes into the imfit 
    '''  
    beam=imfit(imagename=imagename,region=region_init, **kwarg)
    regions = []
    for i in range(beam['results']['nelements']):
        component = 'component' + str(i)
        x_value=beam['results'][component]['shape']\
                 ['direction']['m0']['value']
        y_value=beam['results'][component]['shape']\
                 ['direction']['m1']['value']
        bmaj_value=beam['results'][component]\
                    ['shape']['majoraxis']['value']
        bmin_value=beam['results'][component]['shape']\
                    ['minoraxis']['value']
        pa_value=beam['results'][component]['shape']\
                  ['positionangle']['value']
        x=str(x_value)+'rad'
        y=str(y_value)+'rad'
        bmaj=str(bmaj_value/2.0*ratio)+'arcsec'
        bmin=str(bmin_value/2.0*ratio)+'arcsec'
        pa=str(pa_value)+'deg'
        region='ellipse[['+x+','+y+'],['+bmaj+','+bmin+'],'+pa+']'
        regions.append(region)

    return regions


def beam_to_regions(beam, ratio=2.0):
    '''
    Write the CASA beam object into CASA
    ------
    Parameters:
    beam: CASA beam object
    The beam acquired from task such as imfit
    ratio: float
    The ratio value to be multiplied to the beam object 
    '''   
    regions = []
    for i in range(beam['results']['nelements']):
        component = 'component' + str(i)
        x_value=beam['results'][component]['shape']\
                 ['direction']['m0']['value']
        y_value=beam['results'][component]['shape']\
                 ['direction']['m1']['value']
        bmaj_value=beam['results'][component]\
                    ['shape']['majoraxis']['value']
        bmin_value=beam['results'][component]['shape']\
                    ['minoraxis']['value']
        pa_value=beam['results'][component]['shape']\
                  ['positionangle']['value']
        x=str(x_value)+'rad'
        y=str(y_value)+'rad'
        bmaj=str(bmaj_value/2.0*ratio)+'arcsec'
        bmin=str(bmin_value/2.0*ratio)+'arcsec'
        pa=str(pa_value)+'deg'
        region='ellipse[['+x+','+y+'],['+bmaj+','+bmin+'],'+pa+']'
        regions.append(region)

    return regions

###########################################################
# main program 

# Set the default value of the ratio. 
for band in bands:
    continuum[band]['ratio'] = ratio
    if band in ['band3_p9_multiple', 'band3_p9_1b','HST_Pbeta','HST_Pbeta2']:
        continuum[band]['ratio'] = 2.0

# set the double gaussian fitting
for band in bands:
    continuum[band]['kwarg'] = {}
    if band == 'band3_p9_1b':
        continuum[band]['kwarg']['estimates'] = scriptDir + 'configs/cluster1b_estimates.txt' 
         

bands = ['band 3', 'band3_2016', 'band3_2016_p5', 'band7_2016', 'band7_2016_p5',
        'band3_2016_p5_p9', 'band3_p9_multiple', 'band3_p9_1b']
bands = ['band7_2016_p5']
bands = ['band3_2016_p5']
bands = ['band3_2016_2_p201', 'band7_2016_2_p201']
bands = ['HST_Pbeta']
bands = ['band3_2016_p5', 'band7_2016_p5_band3Ap']
bands = ['HST_Pbeta2']
bands = ['HST_I']
bands =['band7_2016_band3Ap']
bands = ['HST_Pbeta_contsub']

lats_err = []; lons_err = []

for band in bands:
    regions = []
    with open(continuum[band]['region'], 'r') as infile:
        line = infile.readline()
        while line !='':
            line = infile.readline()
            regions.append(line)
    regions.remove('')

    # fit different regions with 2D Gaussian function and decide 
    # the aperture size to measure the flux. 
    fittedRegions = []
    fittedBeams = []
    for i, region in enumerate(regions):
#         rms = continuum[band]['rms'] / continuum[band]['pbcor'][i]
        fittedBeam = imfit(imagename = continuum[band]['imagename'], 
                            region = region,
#                             rms = rms, 
                            **continuum[band]['kwarg'])
        fittedRegion = beam_to_regions(fittedBeam, ratio=continuum[band]['ratio']) 
        fittedRegions += fittedRegion
#         print(fittedBeam['deconvolved']['component0']['shape']['majoraxis'])
#         print(fittedBeam['deconvolved']['component0']['shape']['majoraxiserror'])
#         lats_err.append(fittedBeam['results']['component0']['shape']['direction']['error']['latitude']['value'])
#         lons_err.append(fittedBeam['results']['component0']['shape']['direction']['error']['longitude']['value'])
        
    # Export the regions into a file. 
    with open (continuum[band]['outfile'], 'w') as outfile:
        outfile.write('#CRTFv0 CASA Region Text Format version 0\n')
        for line in fittedRegions:
            outfile.write(line+', color=green\n')


print(lats_err)
print(lons_err)
