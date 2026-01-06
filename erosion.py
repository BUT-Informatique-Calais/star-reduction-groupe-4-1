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

# Utiliser les données en niveaux de gris pour la détection
if data.ndim == 3:
    data_gray: NDArray = np.mean(data, axis=2)
else:
    data_gray: NDArray = data

mean, median, std = sigma_clipped_stats(data_gray, sigma=3.0)
daofind: DAOStarFinder = DAOStarFinder(fwhm=2.0, threshold=0.5 * std)
sources = daofind(data_gray - median)

if sources is not None:
    print(f"{len(sources)} étoiles détectées")
    
    # Créer un masque pour les étoiles avec des rayons adaptés à leur taille
    mask: NDArray[np.uint8] = np.zeros(data_gray.shape, dtype=np.uint8)
    
    # Marquer chaque étoile avec un rayon adapté à sa taille (basé sur sharpness et flux)
    for source in sources:
        x, y = int(source['xcentroid']), int(source['ycentroid'])
        # Rayon adapté à la taille de l'étoile (entre 4 et 12 pixels)
        radius = int(max(4, min(12, source['sharpness'] * 8 + source['peak'] / 2000)))
        cv.circle(mask, (x, y), radius, 255, -1)
    
    print(f"Pixels dans le masque: {np.count_nonzero(mask)}")
    cv.imwrite('./results/star_mask.png', mask)
    
    # Utiliser l'inpainting pour créer une image de fond
    background: NDArray[np.uint8] = cv.inpaint(image, mask, 3, cv.INPAINT_TELEA)
    
    # Adoucir les bords du masque avec un flou gaussien
    mask_float: NDArray[np.float32] = mask.astype(np.float32) / 255.0
    mask_blurred: NDArray[np.float32] = cv.GaussianBlur(mask_float, (11, 11), 3.0)
    
    # Appliquer la formule d'interpolation : I_final = (M × I_eroded) + ((1-M) × I_original)
    if data.ndim == 3:
        # Pour les images couleur, appliquer sur chaque canal
        result: NDArray[np.uint8] = np.zeros_like(image)
        for i in range(3):
            result[:, :, i] = ((1 - mask_blurred) * image[:, :, i] + 
                              mask_blurred * background[:, :, i]).astype(np.uint8)
    else:
        # Pour les images monochromes
        result: NDArray[np.uint8] = ((1 - mask_blurred) * image + 
                                     mask_blurred * background).astype(np.uint8)
    
    # Sauvegarder l'image sans étoiles
    cv.imwrite('./results/stars_removed.png', result, [cv.IMWRITE_PNG_COMPRESSION, 9])