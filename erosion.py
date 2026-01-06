from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from numpy.typing import NDArray
from astropy.table import Table
from cv2.typing import MatLike

from photutils.aperture import CircularAperture

# Open and read the FITS file
fits_file: str = './examples/HorseHead.fits'
hdul: fits.HDUList = fits.open(fits_file)

# Display information about the file
hdul.info()

# Access the data from the primary HDU
data: NDArray = hdul[0].data

# Access header information
header: fits.Header = hdul[0].header

# Handle both monochrome and color images
if data.ndim == 3:
    # Color image - need to transpose to (height, width, channels)
    if data.shape[0] == 3:  # If channels are first: (3, height, width)
        data = np.transpose(data, (1, 2, 0))
    # If already (height, width, 3), no change needed
    
    # Normalize the entire image to [0, 1] for matplotlib
    data_normalized: NDArray = (data - data.min()) / (data.max() - data.min())
    
    # Save the data as a png image (no cmap for color images)
    plt.imsave('./results/original.png', data_normalized)
    
    # Normalize each channel separately to [0, 255] for OpenCV
    image: NDArray[np.uint8] = np.zeros_like(data, dtype='uint8')
    for i in range(data.shape[2]):
        channel: NDArray = data[:, :, i]
        image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
else:
    # Monochrome image
    plt.imsave('./results/original.png', data, cmap='gray')
    
    # Convert to uint8 for OpenCV
    image: NDArray[np.uint8] = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')
    

# Define a kernel for erosion
kernel: NDArray[np.uint8] = np.ones((15, 15), np.uint8)
# Perform erosion
eroded_image: MatLike = cv.erode(image, kernel, iterations=1)

# Save the eroded image 
cv.imwrite('./results/eroded.png', eroded_image)

# Close the file
hdul.close()

# Détecter les étoiles et créer un masque binaire
mean: float
median: float
std: float
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
daofind: DAOStarFinder = DAOStarFinder(fwhm=5.0, threshold=std)
sources = daofind(data - median)

for col in sources.colnames:
    if col not in ('id', 'npix'):
        sources[col].info.format = '%.2f'

mask: NDArray[np.uint8] = np.zeros(data.shape, dtype=np.uint8)

# Créer les apertures circulaires pour marquer les étoiles
positions: NDArray = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures: CircularAperture = CircularAperture(positions, r=4.0)

# Remplir le masque avec les apertures
star_masks: list = apertures.to_mask(method='center')
for star_mask in star_masks:
    if star_mask is not None:
        mask += (star_mask.to_image(data.shape) * 255).astype(np.uint8)

cv.imwrite('./results/star_mask.png', mask)