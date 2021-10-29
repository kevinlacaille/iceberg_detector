#! /usr/bin/python3
"""
A group of functions which aid in the measurements of the change in 
normalized difference vegetation index (NDVI) over time, using 
Planet Labs' PlanetScope 4-Band imagery.

Author
------
Kevin Lacaille

See Also
--------
midpoint
main

References
----------
Planet's notebook on importing & parsing data and measuring NDVI: https://github.com/planetlabs/notebooks/tree/master/jupyter-notebooks/ndvi
"""

import numpy as np
import os

def validate_inputs(args):
    """
    Ensures that the input arguments (data_directory, outout_directory) 
    are valid by checking that the directories exist.

    Parameters:
    -----------
        args : Namespace[str]
               The namespace which contains the input arguments:
               data_directory
               output_directory
    
    Returns:
    --------
        None
    """
    
    # Ensure data directory exists
    if not os.path.isdir(args.data_directory):
        raise Exception("Given data directory not found. Please point to an existing data directory.")

    
    # Ensure output directory exists
    if not os.path.isdir(args.output_directory):
        raise Exception("Given output directory not found. Please point to an existing directory to output to.")


def get_data_filenames(data_directory):
    """
    Returns all file names for the images and their metadata.

    Parameters:
    -----------
        data_directory : str
                         The input path to the PlanetScope 4-Band data.
    
    Returns:
    --------
        image_filenames : List[str]
                         All image file names.
        metadata_filenames : List[str]
                         All metadata file names.
    """
    
    from glob import glob

    # All image file names
    image_filenames = glob(data_directory + "/*AnalyticMS*.tif")

    # All metadate file names
    metadata_filenames = glob(data_directory + "/*AnalyticMS_metadata*.xml")

    # Make sure image and metadata files are loaded
    if not image_filenames:
        raise Exception("Data directory does not seem to contain AnalyticsMS GeoTIFFs.")
    if not metadata_filenames:
        raise Exception("Data directory does not seem to contain AnalyticsMS XML metadata.")

    # Make sure all image files exist
    does_image_exist = [os.path.isfile(file) for file in image_filenames]
    if not any(does_image_exist):
        images_that_dont_exist = image_filenames[np.where(np.array(does_image_exist) == False)[0][0]]
        raise Exception("Cannot find the following files:", images_that_dont_exist)
    
    # Make sure all metadata files exist
    does_metadata_exist = [os.path.isfile(file) for file in image_filenames]
    if not any(does_metadata_exist):
        metadata_that_dont_exist = metadata_filenames[np.where(np.array(does_metadata_exist) == False)[0][0]]
        raise Exception("Cannot find the following files:", metadata_that_dont_exist)
    
    return image_filenames, metadata_filenames


def validate_data(image_filename):
    """
    Ensures that the the given PlanetScope image is a valid image

    Parameters:
    -----------
        image_filename : str
                        The input path to a PlanetScope 4-Band image.
    
    Returns:
    --------
        is_data_valid : bool
                        A flag which returns True if the image is valid.
    """
    import rasterio

    NUM_CHANNELS = 4

    with rasterio.open(image_filename) as src:
        # Check to see if all 4 bands exist
        if len(src.indexes) != NUM_CHANNELS:
            raise Exception("The data does not contain all 4 bands (blue, green, red, nir). Cannot compute NDVI.")
        else:
            src.close()
            is_data_valid = True
            return is_data_valid


def extract_data(image_filename, metadata_filename):
    """
    Extracts un-normalized red and NIR band data from a PlanetScope 4-band image

    Parameters:
    -----------
        image_filename : str
                   The input path to a PlanetScope 4-Band image.
        metadata_filename : str
                   The input path to a PlanetScope 4-Band image metadata.
    
    Returns:
    --------
        band_blue : Array[int]
                   Blue band image.
        band_green : Array[int]
                   Green band image.
        band_red : Array[int]
                   Red band image.
        band_nir : Array[int]
                   NIR band image.
    """
    import rasterio


    # Extract green, red, and NIR data from PlanetScope 4-band imagery
    with rasterio.open(image_filename) as src:
        [band_blue, band_green, band_red, band_nir] = src.read([1,2,3,4])

    return band_blue, band_green, band_red, band_nir


def normalize_data(metadata_filename, band_blue, band_green, band_red, band_nir):
    """
    Normalizes the green, red, and NIR band data values by 
    their reflectance coefficient.

    Parameters:
    -----------
        metadata_filename : str
                   The input path to a PlanetScope 4-Band 
                   image metadata.
        band_green : Array[int]
                   Un-normalized green band image.
        band_red : Array[int]
                   Un-normalized red band image.
        band_nir : Array[int]
                   Un-normalized NIR band image.
    
    Returns:
    --------
        band_green : Array[int]
                   Normalized green band image.
        band_red : Array[int]
                   Normalized red band image.
        band_nir : Array[int]
                   Normalized NIR band image.

    """

    from xml.dom import minidom

    # Parse the XML metadata file
    xmldoc = minidom.parse(metadata_filename)

    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
    
    NUM_CHANNELS = 4

    if nodes.length != NUM_CHANNELS:
        raise Exception("The data does not contain all 4 bands (blue, green, red, nir). Cannot compute NDVI.")

    # XML parser refers to bands by numbers 1-4
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in ['1', '2', '3', '4']:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[i] = float(value)

    # Multiply the Digital Number (DN) values in each band by the TOA reflectance coefficients
    band_blue = band_blue * coeffs[1]
    band_green = band_green * coeffs[2]
    band_red = band_red * coeffs[3]
    band_nir = band_nir * coeffs[4]

    return band_blue, band_green, band_red, band_nir


def measure_ndwi(band_green, band_nir):
    """
    Measures the normalized difference water index (NDWI), 
    defined as: NDVI = (NIR - red) / (NIR + red).

    Parameters:
    -----------
        band_green : Array[int]
               Normalized green band image.
        band_nir : Array[int]
               Normalized NIR band image.
    
    Returns:
    --------
        ndwi : float
               Normalized difference water index

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Normalized_difference_water_index
    """

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Calculate NDVI. This is the equation at the top of this guide expressed in code
    ndwi = (band_green.astype(float) - band_nir.astype(float)) / (band_green + band_nir)

    return ndwi


def apply_water_mask(band_green, band_red, band_nir):
    """
    Uses the normalzied difference water index (NDWI) to
    mask out regions with water.

    Parameters:
    -----------
        band_green : Array[int]
               Normalized green band image.
        band_red : Array[int]
               Normalized red band image.
        band_nir : Array[int]
               Normalized NIR band image.
    
    Returns:
    --------
        band_red : Array[int]
               Normalized red band image, masked from water.
        band_nir : Array[int]
               Normalized NIR band image, masked from water.
    """

    # Measure NWDI
    nwdi = measure_ndwi(band_green, band_nir)

    # Pixel is water if nwdi >= 0.3 (ref: https://www.mdpi.com/2072-4292/5/7/3544/htm)
    WATER_VALUE = 0.3
    water_mask_rows, water_mask_cols = np.where(nwdi >= WATER_VALUE)

    # Apply mask to red and NIR (used for NDVI calculation)
    band_red[water_mask_rows, water_mask_cols] = np.nan
    band_nir[water_mask_rows, water_mask_cols] = np.nan

    return band_red, band_nir


def measure_ndvi(band_red, band_nir):
    """
    Measures the normalized difference vegetation index (NDVI), 
    defined as: NDVI = (NIR - red) / (NIR + red).
    See more: Wikipedia on NDVI: https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index

    Parameters:
    -----------
        band_red : Array[int]
               Normalized red band image.
        band_nir : Array[int]
               Normalized NIR band image.
    
    Returns:
    --------
        ndvi : float
               Normalized difference vegetation index

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index
    """

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Calculate NDVI. This is the equation at the top of this guide expressed in code
    ndvi = (band_nir.astype(float) - band_red.astype(float)) / (band_nir + band_red)

    return ndvi


def measure_ndgi_keshri(band_red, band_green):
    """
    Measures the normalized difference glacier index (NDGI), 
    defined as: NDVI = (NIR - red) / (NIR + red).
    See more: Wikipedia on NDVI: https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index

    Parameters:
    -----------
        band_red : Array[int]
               Normalized red band image.
        band_nir : Array[int]
               Normalized NIR band image.
    
    Returns:
    --------
        ndvi : float
               Normalized difference vegetation index

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index
    """

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Calculate NDVI. This is the equation at the top of this guide expressed in code
    ndgi = (band_green.astype(float) - band_red.astype(float)) / (band_green + band_red)

    return ndgi


def measure_ndgi_lacaille(band_nir, band_green):
    """
    Measures the normalized difference glacier index (NDGI), 
    defined as: NDVI = (NIR - red) / (NIR + red).
    See more: Wikipedia on NDVI: https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index

    Parameters:
    -----------
        band_red : Array[int]
               Normalized red band image.
        band_nir : Array[int]
               Normalized NIR band image.
    
    Returns:
    --------
        ndvi : float
               Normalized difference vegetation index

    See Also:
    ---------
    https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index
    """

    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Calculate NDVI. This is the equation at the top of this guide expressed in code
    ndgi = (band_nir.astype(float) - band_green.astype(float)) / (band_green + band_nir)

    return ndgi    

def compute_rate_of_change(time, ndvi, proportion_dirt, proportion_veg, image_filenames):
    """
    Computes the average rate of change of the NDVI over
    time by measuring the mean change in NDVI per day.

    Parameters:
    -----------
        time : Array[int]
               Number of days from observation
               date until today.
        ndvi : Array[int]
               Normalized difference vegetation
               index per day.
        proportion_dirt : Array[float]
               The proportion of pixels containing 
               dirt to total number of pixels, in
               the image.
        proportion_veg : Array[float]
               The proportion of pixels containing 
               vegetation to total number of pixels,
               in the image.
    
    Returns:
    --------
        NONE
    """

    # Ensure image_filename is just the file name
    image_dates = [image_filename.split("/")[-1].split("_")[0] for image_filename in image_filenames]

    # Find the dates with min and max dirt & veg (assumes only 1 min and max exists)
    min_dirt_date = image_dates[np.where(proportion_dirt == np.min(proportion_dirt))[0][0]]
    min_dirt_date = min_dirt_date[0:4] + "-" + min_dirt_date[4:6] + "-" + min_dirt_date[6:]

    max_dirt_date = image_dates[np.where(proportion_dirt == np.max(proportion_dirt))[0][0]]
    max_dirt_date = max_dirt_date[0:4] + "-" + max_dirt_date[4:6] + "-" + max_dirt_date[6:]

    min_veg_date = image_dates[np.where(proportion_veg == np.min(proportion_veg))[0][0]]
    min_veg_date = min_veg_date[0:4] + "-" + min_veg_date[4:6] + "-" + min_veg_date[6:]

    max_veg_date = image_dates[np.where(proportion_veg == np.max(proportion_veg))[0][0]]
    max_veg_date = max_veg_date[0:4] + "-" + max_veg_date[4:6] + "-" + max_veg_date[6:]

    # Print proportions of dirt and veg over time series
    print("Over the time series, between " + str(round(np.min(proportion_dirt) * 100, 2)) + \
          "% (" + min_dirt_date + ") and " +  str(round(np.max(proportion_dirt) * 100, 2)) + \
          "% (" + max_dirt_date + ") of the region contained barren dirt.")
    print("Over the time series, between " + str(round(np.min(proportion_veg) * 100, 2)) + \
          "% (" + min_veg_date + ") and " +  str(round(np.max(proportion_veg) * 100, 2)) + \
          "% (" + max_veg_date + ") of the region contained vegetation.")

    # Mean change in NDVI and its standard deviation
    mean_change_in_ndvi = np.mean(np.diff(ndvi))
    mean_change_in_ndvi_uncertainty = np.std(np.diff(ndvi))

    # Number of days in time series
    num_days = np.max(time) - np.min(time)

    # Mean rate of change in NDVI and its uncertainty
    mean_rate_of_change = mean_change_in_ndvi / num_days
    mean_rate_of_change_uncertainty = abs((mean_change_in_ndvi_uncertainty / mean_change_in_ndvi)) * abs(mean_rate_of_change)

    # Print rate of change in NDVI per day with uncertainty
    ndvi_result_message = "(" + str(round(mean_rate_of_change * 100, 2)) + " +/- " + \
                           str(round(mean_rate_of_change_uncertainty * 100, 2)) + ") % per day"

    # Print findings - Entire time averaging
    if mean_rate_of_change_uncertainty >= abs(mean_rate_of_change):
        print("Average change in NDVI / day:", ndvi_result_message)
        print("Greenness of the vegetation changed over time, however, over the entire time series, " \
              "the vegetation did not statistically get greener nor less green over time.")
    else:
        if mean_rate_of_change < 0:
            print("Over the entire time series, vegetation is getting less green over time, at a rate of:", ndvi_result_message)
        elif mean_rate_of_change > 0:
            print("Over the entire time series, vegetation is getting more green over time, at a rate of:", ndvi_result_message)
        else:
            print("Over the entire time series, vegetation is neither getting greener nor getting less green over time!")


def visualize_image(image, image_name, image_filename, output_directory):
    """
    This function is modified from: https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/ndvi/ndvi_planetscope.ipynb

    Visualizes a map.

    Parameters:
    -----------
        image : Array[int]
                An image, such as NDVI, Green Band, Red Band, 
                or NIR Band.
        image_name : str
                The name of the image.
        image_filename : str
                The input path to a PlanetScope 4-Band image.
        output_directory : str
                The path to a directory to output figures to.
    
    Returns:
    --------
        NONE
    """

    import matplotlib.pyplot as plt
    from midpoint import MidpointNormalize
    import os

    # Set min/max values from pixel intensity (excluding NAN)
    min_range = np.nanmin(image)
    max_range = np.nanmax(image)
    mid_range = np.mean([min_range, max_range])

    # Ensure image_filename is just the date
    image_date = image_filename.split("/")[-1].split("_")[0]

    # Map
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)

    # diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
    cmap = plt.cm.RdYlGn 

    cax = ax.imshow(image, cmap=cmap, clim=(min_range, max_range), \
                    norm=MidpointNormalize(midpoint=mid_range,vmin=min_range, vmax=max_range))

    ax.axis('off')
    ax.set_title(image_name, fontsize=18, fontweight='bold')
    fig.colorbar(cax, orientation='horizontal', shrink=0.65)

    fig.savefig(os.path.join(output_directory, image_date + "-" + image_name + "-fig.png"), \
                dpi=200, bbox_inches='tight', pad_inches=0.7)
    # plt.show()
    plt.close()


    # Histogram of values in image
    fig2 = plt.figure(figsize=(10,10))
    ax = fig2.add_subplot(111)

    plt.title("NDVI Histogram", fontsize=18, fontweight='bold')
    plt.xlabel("NDVI values", fontsize=14)
    plt.ylabel("# pixels", fontsize=14)

    x = image[~np.isnan(image)]
    NUM_BINS = 20
    ax.hist(x,NUM_BINS,color='green',alpha=0.8)

    # Place a line at the median to help see where the value is coming from
    plt.axvline(x = np.nanmedian(image))

    fig2.savefig(os.path.join(output_directory, image_date + "-" + image_name + "-histogram.png"), \
                 dpi=200, bbox_inches='tight', pad_inches=0.7)
    # plt.show()
    plt.close()


def measure_dirt_veg_proportions(ndvi):
    """
    Measures the proportion of barren dirt pixels and vegegation 
    pixels to the total number of pixels in the image.

    Parameters:
    -----------
        ndvi : Array[float]
               The median NDVI values over the entire region
               at the given acquisition date.
    Returns:
    --------
        proportion_dirt : Array[float]
                Ratio of the total number of pixels containing 
                dirt to the total number of pixels in the scene.
        proportion_veg : Array[float]
                Ratio of the total number of pixels containing 
                vegetation to the total number of pixels in the scene.        
    """    

    # See additional_notes.md for these values
    # Assume 0 <= NDVI <= 0.3 for dirt
    MIN_DIRT_INDEX = 0
    MAX_DIRT_INDEX = 0.3
    # Assume 0.3 < NDVI <= 1.0 for vegetation
    MIN_VEG_INDEX = 0.3

    # Remove all nans for the future calculations
    ndvi = ndvi[~np.isnan(ndvi)]

    # Total number of pixels in the scene
    num_pixels = np.size(ndvi)

    # Compute ratios
    proportion_dirt = len(np.where((ndvi >= MIN_DIRT_INDEX) & (ndvi <= MAX_DIRT_INDEX))[0]) / float(num_pixels)
    proportion_veg = len(np.where(ndvi > MIN_VEG_INDEX)[0]) / float(num_pixels)

    return proportion_dirt, proportion_veg


def visualize_data(time, ndvi, proportion_dirt, proportion_veg, output_directory):
    """
    Visualizes how the NDVI changes over time.

    Parameters:
    -----------
        time : Array[float]
               The time from today to the acquisition date.
        ndvi : Array[float]
               The median NDVI values over the entire region
               at the given acquisition date.
        proportion_dirt : Array[float]
                Time series ratio of the total number of pixels containing 
                dirt to the total number of pixels in the scene.
        proportion_veg : Array[float]
                Time series ratio of the total number of pixels containing 
                vegetation to the total number of pixels in the scene.        
        output_directory : str
                The path to a directory to output figures to.
    
    Returns:
    --------
        NONE
    """

    mean_ndvi = np.mean(ndvi)
    mean_change_in_ndvi = np.mean(np.diff(ndvi))
    mean_change_in_ndvi_uncertainty = np.std(np.diff(ndvi))

    import matplotlib.pyplot as plt

    # Normalize time by initial acquisition
    time = time - np.min(time)

    # NDVI vs. time
    fig3 = plt.figure()
    fig3.add_subplot(111)

    plt.title("NDVI time series", fontsize=18, fontweight='bold')
    plt.xlabel("Time (days from initial acquisition)", fontsize=14)
    plt.ylabel("Median NDVI values", fontsize=14)

    plt.plot(time, ndvi, label='_nolegend_')

    # The average NDVI over time
    plt.plot(time, mean_change_in_ndvi * np.ones(len(time)) + mean_ndvi, "k-")
    plt.plot(time, (mean_change_in_ndvi + mean_change_in_ndvi_uncertainty)* np.ones(len(time)) + mean_ndvi, "g--")
    plt.plot(time, (mean_change_in_ndvi - mean_change_in_ndvi_uncertainty) * np.ones(len(time)) + mean_ndvi, "r--")

    plt.legend(["Mean", r"+1 $\sigma$", r"-1 $\sigma$"])
    fig3.savefig(output_directory + "/temporal-NDVI.png", dpi=200, bbox_inches='tight', pad_inches=0.7)
    # plt.show()


    # Proportions of dirt & veg vs. time
    fig4 = plt.figure()
    fig4.add_subplot(111)

    plt.title("Proportions of dirt and vegetation", fontsize=18, fontweight='bold')
    plt.xlabel("Time (days from initial acquisition)", fontsize=14)
    plt.ylabel("Number of px / total px", fontsize=14)

    plt.plot(time, proportion_dirt, 'brown')
    plt.plot(time, proportion_veg, 'green')

    plt.legend(["Dirt", "Vegetation"])

    fig4.savefig(output_directory + "/temporal-dirt-veg-proportions.png", dpi=200, bbox_inches='tight', pad_inches=0.7)
    # plt.show()