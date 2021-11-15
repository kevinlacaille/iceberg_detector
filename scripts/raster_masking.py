"""Masking a raster with a vector.

CRS = Coordinate Reference System

The general idea is that the vector must be burned into a raster that can then
be used as a mask in Numpy. If the vector and image are in the same CRS then
this is pretty easy. Different CRS require reprojecting one to the other, which
can be difficult depending on the CRS involved and the size and density of the
geometries.

There are of course multiple ways to do this. See the 'straightforward()'
function for a method that is easier to follow and illustrates the core
concepts. Although it is so slow that I didn't let it finish and don't actually
know if it works. The 'optimized()' function does some trickery to reduce CPU
time and memory footprint, and is closer to what I would implement in a
production system where these things are necessary. Elements of can also be
combined. The specific geometries and locations of scenes I am likely to see
might force me to implement some of the other things mentioned below.

Warping a vector is inherently an imprecise operation unless the vector is
densified with extra points. I think you're far enough away from the edge of
the world to generally not experience these problems, but the GSHHG geometries
are so large that reprojecting some points is impossible because they fall
outside the target projection's bounds. There are tricks for dealing with this
case, but all have tradeoffs. You could rasterize the geometries in the source
CRS and reproject to the target CRS and deal with some ambiguity along the
edges of the mask when applying it to the target scene. The source geometries
could be cut up into little chunks to prevent a point on the other side of the
world from interfering with the small region that you want to warp. The bounding
box of the scene could be reprojected to the vector CRS, probably dilated a bit,
used to clip out a chunk of the vector, reproject that chunk to the image CRS,
and then rasterize.
"""


from contextlib import contextmanager

import fiona as fio
import numpy as np
import rasterio as rio
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import transform_geom
from shapely.geometry import CAP_STYLE, JOIN_STYLE, box, mapping, shape


gshhg_path = 'gshhg_global_land/gshhg_land.shp'
scene_path = 'lacaille/20210629_132002_13_245a_3B_AnalyticMS_DN.tif'
outfile = '20210629_132002_13_245a-masked-ndsi.tif'
creation_options = {
    'TILED': 'YES',
    'PREDICTOR': '3',
    'COMPRESS': 'DEFLATE'
}


@contextmanager
def seterr(**kwargs):

    """Never versions of Numpy let 'np.seterr()' operate as a context manager,
    but the pipeline is on 3.5.
    """

    oldstate = np.seterr(**kwargs)
    try:
        yield
    finally:
        np.seterr(**oldstate)


def straightforward():

    """This implementation is so slow I never actually let it finish and have
    no idea if it works. Slowness is caused by not filtering input geometries
    and instead blindly reprojecting all and possibly attempting to rasterize
    all.
    """

    with rio.open(scene_path) as src_img, fio.open(gshhg_path) as src_gshhg:

        # Reproject all geometries to the scene's CRS
        geometries = (f['geometry'] for f in src_gshhg)
        geometries = (
            transform_geom(
                src_crs=src_img.crs,
                dst_crs=src_img.crs,
                geom=g
            ) for g in geometries)

        # Generate a 2D Numpy array where True elements are masked and False
        # are not. False elements are water and True are land. Might be wise
        # to erode the water pixels a bit to avoid identifying snow/ice on the
        # edge of the water as icebergs. Each element in this array maps
        # directly to a pixel in the image, so eroding/dilating by 1 pixel can
        # be translated to a physical distance on the ground.
        gshhg_mask = geometry_mask(
            geometries=geometries,
            out_shape=(src_img.height, src_img.width),
            transform=src_img.transform,
            all_touched=False)

        # For Planetscope scenes this mask identifies the "image collar", AKA
        # the blackfill region around the frame. This can also be identified
        # with the UDM. An argument could be made for constructing a mask for
        # only the blue and NIR bands, but in this case using the entire
        # dataset mask is fine and ultimately identical to the mask for only
        # the 2 bands. Since the image collar is entirely 0 this is also kind
        # of inherently present during the NDSI calculation. You might also
        # want to incorporate some bits from the UDM here, especially the
        # saturated pixel mask. I noticed that '20210629_132002_13_245a' has
        # some blooming pixels, which are radiometrically invalid.
        mask = src_img.dataset_mask()

        # Combine the dataset mask with the GSHHG mask. Band math requires
        # valid pixels in both bands, so one mask can be applied to both.
        mask |= gshhg_mask

        # Read only the necessary bands.
        blue, nir = src_img.read(indexes=(1, 4), out_dtype=np.float32)

        # Apply the mask.
        blue[mask] = np.nan
        nir[mask] = np.nan

        # Calculate NDSI
        with seterr(divide='ignore', invalid='ignore'):
            ndsi = (blue - nir) / (blue + nir)

        height, width = ndsi.shape
        profile = {
            'count': 1,
            'height': height,
            'width': width,
            'dtype': rio.float32,
            'driver': 'GTiff',
            'crs': src_img.crs,
            'transform': src_img.transform,
            'nodata': float('nan'),
            **creation_options
        }

        with rio.open(outfile, 'w', **profile) as dst:
            dst.write(ndsi, 1)


def optimized():

    """Tricks:

        1. Get the scene's bounding box.
        2. Dilate the geometry by the longest edge of the box.
        3. Reproject this geometry to the vector's CRS.
        4. Apply as a spatial filter to avoid having to avoid even reading
           geometries that do not intersect.
        5. Clip vector geometries against the scene's buffered bounding box
           to drastically reduce the amount of points that have to be warped.

    This is probably about as fast as it gets without using a sparser dataset,
    simplifying the input geometries, or tiling the source data to work with
    little chips rather than geometries the size of a continent.
    """

    with rio.open(scene_path) as src_img, fio.open(gshhg_path) as src_gshhg:

        # The image's bounding box can be reprojected to the vector CRS for
        # use as a spatial filter to prevent warping geometries that do not
        # actually intersect the scene. Since the GSHHG geometries are so big
        # this geometry will also be used to clip geometries.
        gshhg_filter_geom = box(*src_img.bounds)
        xmin, ymin, xmax, ymax = gshhg_filter_geom.bounds
        buffer_distance = max((xmax - xmin, ymax - ymin))
        gshhg_filter_geom.buffer(
            buffer_distance,
            cap_style=CAP_STYLE.flat,
            join_style=JOIN_STYLE.mitre)

        # Reproject from scene CRS to GSHHG
        gshhg_filter_geom = transform_geom(
            src_crs=src_img.crs,
            dst_crs=src_gshhg.crs,
            geom=mapping(box(*src_img.bounds)))

        # Only read geometries intersecting the scene's footprint.
        geometries = (
            f['geometry']
            for f in src_gshhg.filter(mask=gshhg_filter_geom))

        # Clip vector layer to the same geometry used as a spatial filter.
        shapely_gshhg_filter_geom = shape(gshhg_filter_geom)
        geometries = (
            shape(g).intersection(shapely_gshhg_filter_geom)
            for g in geometries)

        # Convert geometries from Shapely objects to Python dictionaries
        # representing GeoJSON
        geometries = (mapping(g) for g in geometries)

        # Reproject all geometries to the scene's CRS
        geometries = (
            transform_geom(
                src_crs=src_gshhg.crs,
                dst_crs=src_img.crs,
                geom=g
            ) for g in geometries)

        # The GSHHG geometry represents land, which we want to be masked.
        # GDAL doesn't have a boolean datatype, so just use 0's for unmasked
        # and 1's for masked, which can be converted to a Numpy boolean array
        # later.
        mask = rasterize(
            shapes=geometries,
            out_shape=(src_img.height, src_img.width),
            fill=0,
            transform=src_img.transform,
            all_touched=True,
            default_value=1,
            dtype=np.uint8,
        )

        # There are other tricks to avoid this array copy, but they are fairly
        # obtuse.
        mask = mask.astype(bool)

        blue, nir = src_img.read(indexes=(1, 4), out_dtype=np.float32)

        # Combine with the dataset mask. Rasterio's 'dataset_mask()' is
        # inefficient by nature for images with a nodata value and effectively
        # causes us to do two reads. For the sake of usability I would stick
        # with it anyway since GDAL's mask flags are complicated unless I really
        # needed the extra performance.
        #   https://gdal.org/development/rfc/rfc15_nodatabitmask.html
        #   https://rasterio.readthedocs.io/en/latest/topics/masks.html
        mask |= blue == 0
        mask |= nir == 0

        # Read only the necessary bands.
        blue, nir = src_img.read(indexes=(1, 4), out_dtype=np.float32)

        # Apply the mask
        blue[mask] = np.nan
        nir[mask] = np.nan

        # Calculate NDSI
        with seterr(divide='ignore', invalid='ignore'):
            ndsi = (blue - nir) / (blue + nir)

        height, width = ndsi.shape
        profile = {
            'count': 1,
            'height': height,
            'width': width,
            'dtype': rio.float32,
            'driver': 'GTiff',
            'crs': src_img.crs,
            'transform': src_img.transform,
            'nodata': float('nan'),
            **creation_options
        }

        with rio.open(outfile, 'w', **profile) as dst:
            dst.write(ndsi, 1)