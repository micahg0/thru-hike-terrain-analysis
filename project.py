import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import svd
from scipy.io import wavfile
from scipy import signal
from scipy.spatial.distance import pdist, squareform
import itertools
from pyproj import Proj, transform, Transformer
from pathlib import Path
import pywt
import subprocess
import os
import requests
from osgeo import gdal

import geopandas as gpd
import shapely
import geopy



logger = logging.getLogger()


class Trail:
    """ Class for working with data along trail, including shapefiles, Digital Elevation Model (DEM) files, and downloading from OpenTopography.com API"""

    def __init__(self):
        self.shapefile_directory = None
        self.df = None
        self.coordinates = None
        self.file_path = None
        self.idx_start = None
        self.idx_stop = None
        self.coordinates_sampled = None
        self.bounding_box_set = None
        self.bounding_box_centers = None
        return
    
    def process_trail_from_shapefile(self, shapefile_directory, subsample_rate, spacing):
        """Script to perform common actions, including shapefile, get coordinates, subsample, and sort"""

        self.load_shapefile_to_dataframe(shapefile_directory=shapefile_directory)
        self.get_coordinates_from_dataframe()
        self.subsample_coordinates(subsample_rate=subsample_rate)
        self.sort_coordinates()
        self.filter_coordinates_with_spacing(spacing=spacing)
        self.get_mapping_bounding_boxes()
        self.get_dem_bounding_boxes()


        return

    def load_shapefile_to_dataframe(self, shapefile_directory):
        """Load shapefile using `geopandas` and produce dataframe"""

        self.shapefile_directory = shapefile_directory
        self.df = gpd.read_file(self.shapefile_directory)
        logging.info('Loaded shapefile from {} with shape {}.'.format(self.shapefile_directory, self.df.shape))
        self.df = self.df[-self.df.geometry.isnull()]
        logging.info('After removing any null geometry rows, df has shape {}'.format(self.df.shape))

        return
    
    def get_coordinates_from_dataframe(self, map_crs="EPSG:3857"):
        """Extracts coordinates from dataframe. Defaults to using Web Mercator Format (EPSG:3857)"""
        self.df = self.df.to_crs(map_crs)
        x = []
        y = []
        for geometry in self.df.geometry:
        
            if geometry.geom_type=='LineString':
                for coordinate in geometry.coords:
                    x.append(coordinate[0])
                    y.append(coordinate[1])
            elif geometry.geom_type=='MultiLineString':
                for linestring in geometry:
                    for coordinate in linestring.coords:
                        x.append(coordinate[0])
                        y.append(coordinate[1])
        

        self.coordinates = np.vstack((x, y))
        logging.info('Extracted coordinates with shape {} from dataframe geomerty column (LineStrings and MultiLineStrings)'.format(self.coordinates.shape))
        return
    
    def subsample_coordinates(self, subsample_rate=100):
        """ Samples a fraction of the full coordinate list. Defaults to sampling 1/100 samples"""

        self.coordinates = self.coordinates[:,::subsample_rate]
        logging.info('Coordinates reduced by factor of subsample rate {}, new coordinates shape {}'.format(subsample_rate, self.coordinates.shape))
        return

    def sort_coordinates(self):
        """Sort coordinates based on most northern and most southern parts of the trails (the beginning and ends)"""
        y_min = self.coordinates[1].min()
        y_max = self.coordinates[1].max()
        logging.info('Latitude (or y) min: {}; max: {}'.format(y_min, y_max))
        self.idx_start = self.coordinates[1].argmin()
        self.idx_stop = self.coordinates[1].argmax()
        
        logging.info('Starting coodinate (most southern) {} at index {}'.format(self.coordinates[:,self.idx_start], self.idx_start))
        logging.info('Stopping coodinate (most northern) {} at index {}'.format(self.coordinates[:,self.idx_stop], self.idx_stop))

        self.coordinates = self.find_gps_sorted(self.coordinates.T, k0=self.idx_start, kend=self.idx_stop)
        # coordinates = coords_sorted.T.reshape(2,-1)
        logging.info('Coordinates sorted and trimmed into shape {}'.format(self.coordinates.shape))
        self.coordinates_sampled = self.coordinates

        return 


    def save_trail_coordinates(self, file_path):
        """Save trail coordinates to csv. Useful after processing and before analysis"""

        self.file_path = file_path
        np.savetxt(file_path, self.coordinates)
        return

    def filter_coordinates_with_spacing(self, spacing):
        """Chooses coordinates separated by specific spacing. Allows no overlap of test points"""

        self.spacing=spacing
        self.bounding_box_centers = []
        self.bounding_box_set = []
        logging.info('Filtering {} coordinates with spacing {}'.format(self.coordinates.shape, spacing))
        for coord in self.coordinates.T:
            self.check_in_bounding_box_set(coord)
            
            if not self.in_bounding_box_set:
                bounding_box = self.get_bounding_box_coordinates(coord, self.spacing)
                self.bounding_box_set.append(bounding_box)
                logging.debug('Added bounding box {} to bounding box set (now has {} bounding boxes)'.format(bounding_box, len(self.bounding_box_set)))
                bounding_box_center = (coord[0], coord[1])
                self.bounding_box_centers.append(bounding_box_center)
                logging.debug('Added bounding box center {} to bounding box centers (now has {} bounding boxe centers)'.format(bounding_box_center, len(self.bounding_box_centers)))

        self.coordinates = np.array(self.bounding_box_centers).T
        logging.info('Coordinates shape filtered with spacing {}; new shape {}'.format(self.spacing, self.coordinates.shape))
        return


    def check_in_bounding_box_set(self, coord):
        """Checks if the coordinate is in the current bounding box set. Returns True or False"""

        logging.debug('Checking if {} is in bounding box set'.format(coord))
        self.in_bounding_box_set = False
        for bbox in self.bounding_box_set:
            if coord[0] < bbox[0][0] and coord[0] > bbox[1][0] and coord[1] < bbox[0][1] and coord[1] > bbox[1][1]:
                self.in_bounding_box_set = True
        
        logging.debug('Coordinate {} is set: {}'.format(coord, self.in_bounding_box_set))

        return self.in_bounding_box_set

    def get_mapping_bounding_boxes(self):
        """Retreives bounding boxes for use in plotting coordinates on web maps, specifically in Bokeh"""

        north = self.coordinates[1] + self.spacing
        south = self.coordinates[1] - self.spacing

        east = self.coordinates[0] + self.spacing
        west = self.coordinates[0] - self.spacing
        
        self.mapping_bounding_boxes = np.vstack((west, east, south, north))
        logging.info('Calculated bounding boxes for plotting on web maps. First sample {}; shape {}'.format(self.mapping_bounding_boxes[:,0], self.mapping_bounding_boxes.shape))
        return

    def get_dem_bounding_boxes(self):
        """Retreives bounding boxes for use in downloading Digital Elevation Model (DEM) files from OpenTopography.com"""

        inProj = Proj('epsg:3857')
        outProj = Proj('epsg:4326')
        self.spacing_degrees = self.spacing/111320
        coord_degrees = transform(inProj, outProj, self.coordinates[0], self.coordinates[1], always_xy=True)
        
        wgs_north = coord_degrees[1] + self.spacing_degrees
        wgs_south = coord_degrees[1] - self.spacing_degrees

        wgs_east = coord_degrees[0] + self.spacing_degrees
        wgs_west = coord_degrees[0] - self.spacing_degrees

        self.dem_bounding_boxes = np.vstack((wgs_west, wgs_east, wgs_south, wgs_north))
        logging.info('Calculated bounding boxes for download DEM files. First sample {}; shape {}'.format(self.dem_bounding_boxes[:,0], self.dem_bounding_boxes.shape))
        logging.debug('east-west:\n{}\north-south:\n{}'.format(wgs_east-wgs_west, wgs_north-wgs_south))

        return

    def download_coordinates(self, relative_directory):
        """Downloads multiple Digital Elevation Model (DEM) files coordinate list"""


        self.relative_directory = relative_directory
        logging.info('Starting to download {} DEM files to {}'.format(self.dem_bounding_boxes.shape, self.relative_directory))
        for i in range(0, self.dem_bounding_boxes.shape[1]):
            bbox = self.dem_bounding_boxes[:,i]
            file_name = 'out-{:03d}.img'.format(i)
            file_path = os.path.join(self.relative_directory, file_name)
            self.download_dem_from_opentopography(bbox=bbox, file_path=file_path)
            logging.info('Downloaded {} to {}'.format(i, file_path))
        return

    @staticmethod
    def download_dem_from_opentopography(bbox, file_path):
        """Downloads individual Digital Elevation Model (DEM) file from OpenTopography.com API"""

        demtype='SRTMGL3'
        URL = 'https://portal.opentopography.org/otr/getdem'
        outputFormat = 'IMG'
        west = bbox[0]
        east = bbox[1]
        south = bbox[2]
        north = bbox[3]
        logging.info('Downloading bounding box {}'.format(bbox))
        logging.info('West: {}, Easth {}, South {}, North {}'.format(west, east, south, north))
        PARAMS={'demtype': demtype,
                'west': west,
                'east': east,
                'south': south,
                'north': north}
        r = requests.get(url = URL, params = PARAMS)
        logging.info('Response header content type: {}; Code {}'.format(r.headers['Content-Type'], r.status_code))
        with open(file_path, 'wb') as f:
            for block in r.iter_content(1024):
                f.write(block)
        logging.info('Downloaded DEM of {} to file {}'.format(bbox, file_path))                    
        return 


    @staticmethod
    def get_bounding_box_coordinates(coord, spacing):
        """Returns bounding box spacing around coordinate"""

        distance = spacing*2
        north = coord[1] + distance
        south = coord[1] - distance

        east = coord[0] + distance
        west = coord[0] - distance

        bounding_box_coordinates = [(east, north), (west, south)] 
        logging.debug('Bounding box coordinates returned: {}'.format(bounding_box_coordinates))
        
        return bounding_box_coordinates



    @staticmethod
    def find_gps_sorted(xy_coord, k0=0, kend=-1):
        """Find iteratively a continuous path from the given points xy_coord,
        starting by the point indexes by k0;
        modified from https://stackoverflow.com/questions/31456683/plot-line-from-gps-points""" 


        N = len(xy_coord)
        distance_matrix = squareform(pdist(xy_coord, metric='euclidean'))
        mask = np.ones(N, dtype='bool')
        sorted_order = np.zeros(N, dtype=np.int)
        indices = np.arange(N)

        i = 0
        k = k0
        while True:
            if k==kend:
                break
            sorted_order[i] = k
            mask[k] = False

            dist_k = distance_matrix[k][mask]
            
        
            indices_k = indices[mask]

            if not len(indices_k):
                break

            # find next unused closest point
            k = indices_k[np.argmin(dist_k)]

            # you could also add some criterion here on the direction between consecutive points etc.
            i += 1
        #drop last point
        coords_sorted = xy_coord[sorted_order]
        # trim the last setments to remove issues
        coords_sorted= coords_sorted[:-10]
        return coords_sorted.T

class Terrain:
    """ Class for working with data along trail, including shapefiles, Digital Elevation Model (DEM) files, and downloading from OpenTopography.com API"""

    def __init__(self):
        self.dem_directory = None
        self.coordinates = None
        return
    
    def load_dem_directory(self, dem_directory, file_limit=False, dem_size = 431*431):
        """Loads Digital Elevation Model (DEM) files from directory into array"""
        self.file_limit = file_limit
        self.dem_size = dem_size
        self.dem_directory = dem_directory
        self.num_files = len(os.listdir(dem_directory))
        logging.info('DEM directory has {} files'.format(self.num_files))
        if self.file_limit is False:
            self.X = np.zeros((self.num_files, self.dem_size))
            dem_files = Path(self.dem_directory).rglob('*.img')
            logging.info('Loading DEM files with size {} from {} with no file limit'.format(self.dem_size, self.dem_directory))
        else:
            self.X = np.zeros((self.file_limit, self.dem_size))
            dem_files = itertools.islice(Path(self.dem_directory).rglob('*.img'), self.file_limit)
        i=0
        for dem_path in dem_files:
            geo = gdal.Open(str(dem_path))
            arr = geo.ReadAsArray()
            logging.debug('Add array with shape {}'.format(arr.shape))
            arr = np.array(arr).flatten()
            self.X[i] = arr
            logging.debug('Processed {} image from from {}'.format(i, dem_path))
            i+=1

        logging.info('Load files from {} to array with shape {}'.format(self.dem_directory, self.X.shape))

        return

    def subtract_mean_from_dem(self):
        """Zeros out the mean from the Digital Elevation Model (DEM) data to remove impact of specific elevation"""

        self.mean_elevations = self.X.mean(axis=1)
        self.X = self.X - self.mean_elevations[:, None]

        return
    
    def get_fft_before_filter(self, sampling_period=30):
        """Return FFT based on Digital Elevation Model (DEM) data before filtering"""

        self.n = int(np.sqrt(self.dem_size))
        # self.Xf = np.copy(self.X)
        # self.Xfs = np.copy(self.X)

        # logging.info('Copy X to Xf with shape {}'.format(self.Xf.shape))
        # for i in range(0, self.Xf.shape[0]):
        #     Xf_ = np.fft.fft2(self.X[i].reshape(self.n, self.n))
        #     self.Xf[i] = Xf_.flatten()
        #     self.Xfs[i] = np.fft.fftshift(Xf_).flatten()
        self.Xf = np.fft.fft(self.X, axis=1)
        self.Xfs = np.fft.fftshift(self.Xf)
        self.f = np.fft.fftfreq(n=self.n, d=30)
        self.fs = np.fft.fftshift(self.f)
        logging.info('Created Xfs with shape {}'.format(self.Xfs.shape))

        ax1_mean = self.Xfs.reshape(-1, 431,431).max(axis=1)
        ax2_mean = self.Xfs.reshape(-1, 431,431).max(axis=2)

        self.Xfs_comp = (ax1_mean + ax2_mean)/2
        logging.info('Created Xfs_comp with shape {}'.format(self.Xfs_comp.shape))

        return

    def get_cwt(self):
        """Return Continuous Wavelet Transform (CWT) cooefficients on Digital Elevation Model (DEM) using Mexican Hat Wavelet"""

        self.cA, self.cD = pywt.cwt(self.X, wavelet='mexh', scales=np.arange(1,15))

        return
    

    def get_fft_after_filter(self, component=5):
        self.n = int(np.sqrt(self.dem_size))

        self.Cf = np.fft.fftn(self.cA[component], axes=[1])
        self.Cfs_comp = []
        for i in range(0, self.Cf.shape[0]):
            arr = self.Cf[i]
            arr_reshape = np.fft.fftshift(np.real(arr.reshape(self.n, self.n)))
            arr_mean1 = arr_reshape.max(axis=0)
            arr_mean2 = arr_reshape.max(axis=1)
            arr_mean = (arr_mean1 +  arr_mean2)/2
            self.Cfs_comp.append(arr_mean)
            # ax2.reshape(-1, 431,431).max(axis=1)
            # ax2_mean = self.Cfs.reshape(-1, 431,431).max(axis=2)
        logging.info('Created Cfs_comp with length {}'.format(len(self.Cfs_comp)))
        self.Cfs_comp = np.array(self.Cfs_comp)
        return


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import glob
    import os
    import sys
    logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    logging.info(os.getcwd())
    os.chdir('./AMATH-582/project/')
    logging.info(os.getcwd())

    os.system('mkdir -p ./test/trail-shapefiles')
    # Unzip and prepare shapefile directories
    # !mkdir -p ./test/trail-shapefiles
    # !unzip -o -q ./data/imported/stelprdb5332131.zip -d ./test/trail-shapefiles/pct/
    # !unzip -o -q ./data/imported/CDTbyState20161128.zip -d ./test/trail-shapefiles/cdt/
    # !unzip -o -q ./data/imported/AT_Centerline_12-23-2014.zip -d ./test/trail-shapefiles/at/

    shapefile_directory = './test/trail-shapefiles/pct'


    # shapefile_directory = './local/trails/CDTbyState20161128'
    # shapefile_directory = './local/trails/AT_Centerline_12-23-2014'


    # trail = Trail()
    # trail.load_shapefile_to_dataframe(shapefile_directory=shapefile_directory)
    # # trail.get_coordinates_from_dataframe(map_crs="EPSG:4326") 
    # trail.get_coordinates_from_dataframe()
    # trail.subsample_coordinates()
    # trail.sort_coordinates()
    # trail.save_trail_coordinates('./test-coords.csv')
    # plt.plot(trail.coordinates[0], trail.coordinates[1])
    # plt.savefig('./test-img.png')

    # trail.filter_coordinates_with_spacing(spacing=1000)
    # # trail.filter_coordinates_with_spacing(spacing=20000)

    # trail.get_mapping_bounding_boxes()
    # trail.get_dem_bounding_boxes()


    # test_bbox = trail.dem_bounding_boxes[:,1]
    # trail.download_dem_from_opentopography(bbox=test_bbox, file_path='./test.img')
    # trail.dem_bounding_boxes = trail.dem_bounding_boxes[:,:3]

    # trail.download_coordinates(relative_directory='./local/test-dem/')
    # trail2 = Trail()
    # trail2.process_trail_from_shapefile('./local/trails/CDTbyState20161128', subsample_rate=1000, spacing=20e3)
    # trail2.dem_bounding_boxes = trail2.dem_bounding_boxes[:,:10]
    # Path('./test/dem/cdt-20000/').mkdir(parents=True, exist_ok=True)

    # trail2.download_coordinates('./test/dem/cdt-20000/')

    ter = Terrain()
    
    ter.load_dem_directory('./test/dem/cdt-20000/', file_limit=False)
    ter.load_dem_directory('./test/dem/cdt-20000/', file_limit = 20)
    ter.subtract_mean_from_dem()
    ter.get_fft_before_filter()
    ter.get_cwt()

