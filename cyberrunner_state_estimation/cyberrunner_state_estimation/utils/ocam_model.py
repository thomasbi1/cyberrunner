import numpy as np
from numpy.polynomial.polynomial import polyval, polyroots

# from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import cv2


class OcamModel:
    """Omnidirectional camera model ported from the Omnidirectional Camera
    Calibration Toolbox for Matlab
    (https://sites.google.com/site/scarabotix/ocamcalib-toolbox). This includes
    only the functionality after the calibration has been completed within
    MATLAB, i.e. parsing of calibration results and 3D world to pixel
    projections and vice-versa. For a complete description of the functions
    refer to the above website.

    """

    def __init__(self, calib_file):
        """Load all camera model parameters from file exported from the MATLAB
        toolbox.

        Parameters
        ----------
        calib_file : string
            Path to the file containing the calibration results.

        """
        with open(calib_file) as f:
            calib_txt = f.read()
        calib_txt = calib_txt.splitlines()

        self.ss = np.fromstring(calib_txt[2], sep=" ")[1:]
        self.inv_ss = np.fromstring(calib_txt[6], sep=" ")[1:]
        self.xc, self.yc = np.fromstring(calib_txt[10], sep=" ")
        self.c, self.d, self.e = np.fromstring(calib_txt[14], sep=" ")
        self.a = np.array([[self.c, self.d], [self.e, 1.0]])
        self.a_inv = np.linalg.inv(self.a)
        self.height, self.width = np.fromstring(calib_txt[18], sep=" ")

        # Edge vectors where z = -1
        self.edges = np.empty((4, 3))
        i = 0
        for u in (0, self.height - 1):
            for v in (0, self.width - 1):
                self.edges[i] = self.cam2world([u, v])
                i += 1
        self.edges /= np.abs(self.edges[:, 2, None])

        self.map1 = None
        self.map2 = None

    def scale(self, factor):
        """Scale down parameters by a given factor.

        Parameters
        ----------
        factor : float
            Factor to scale by. A factor of e.g. 2 halfs the camera's assumed resolution.

        """
        for i in range(5):
            self.ss[i] *= factor ** (i - 1)
        self.inv_ss = self.inv_ss / factor
        self.xc = self.xc / factor
        self.yc = self.yc / factor
        self.height = int(self.height / factor)
        self.width = int(self.width / factor)

    def world2cam(self, world_points, algorithm="invpoly"):
        """Convert 3D world coordinates into 2D pixel coordinates.

        Parameters
        ----------
        world_points : array_like, shape (N, 3) or (3,)
            Each row is a 3D point in (x, y, z) format.
        algorithm : str, optional
            * 'invpoly' : Fast approximation using the inverse polynomial. At
              least 1000 times faster than 'polyroot'.
            * 'polyroot' : Exact solution obtained by finding the root of the
              original polynomial.
            Default is 'invpoly'.
        Returns
        -------
        img_points : numpy.ndarray, shape (N, 2) or (2,)
            Each row is a 2D point in (u, v) format.

        """
        world_points = np.asarray(world_points)
        num_points = world_points.shape[0]

        is1d = world_points.ndim == 1
        if is1d:
            world_points = np.reshape(world_points, (-1, 3))

        old_settings = np.seterr(divide="ignore")

        # Normalize world points with x-y Euclidian distance
        norms = np.linalg.norm(world_points[:, 0:2], axis=1)
        nonzero_idx = norms != 0
        world_points_norm = world_points / norms.reshape(-1, 1)

        np.seterr(**old_settings)

        # Calculate rho according to the specified algorithm
        if algorithm == "invpoly":
            theta = np.arctan(world_points_norm[:, 2])
            rho = polyval(theta, self.inv_ss)
        elif algorithm == "polyroot":
            ss = np.repeat(self.ss.reshape(1, -1), num_points, axis=0)
            ss[:, 1] -= world_points_norm[:, 2]
            rho = np.apply_along_axis(self._minrealpolyroot, 1, ss)
        else:
            raise ValueError("Invalid algorithm '%s'" % algorithm)

        # Calculate image points from rho and apply transformation
        a = np.array([[self.c, self.d], [self.e, 1.0]])
        img_points = world_points_norm[:, 0:2] * rho.reshape(-1, 1)
        img_points = np.einsum("ijk,ik->ij", a[None, :, :], img_points) + np.array(
            [self.xc, self.yc]
        )
        img_points[~nonzero_idx] = np.array([self.xc, self.yc])

        if is1d:
            img_points = img_points.reshape((-1,))

        return img_points

    def cam2world(self, img_points):
        """Converts 2D pixel coordinate into a 3D vector emanating from the
        single effective viewpoint of the camera.

        Parameters
        ----------
        img_points : array_like, shape (N, 2) or (2,)
            Each row is a 2D point in (u, v) format.
        Returns
        -------
        world_points : numpy.ndarray, shape (N, 3) or (3,)
            Each row is a 3D point in (x, y, z) format.

        """
        img_points = np.asarray(img_points)
        is1d = img_points.ndim == 1
        if is1d:
            img_points = np.reshape(img_points, (-1, 2))

        world_points = np.empty((img_points.shape[0], 3))
        world_points[:, :2] = (
            self.a_inv @ (img_points - np.array([self.xc, self.yc])).T
        ).T
        r = np.linalg.norm(world_points[:, :2], axis=1)
        world_points[:, 2] = polyval(r, self.ss)  # z

        # Normalize
        world_points /= np.linalg.norm(world_points, axis=1, keepdims=True)

        if is1d:
            world_points = world_points.reshape((-1,))

        return world_points

    def set_maps(self, map1, map2):
        self.map1 = map1
        self.map2 = map2

    def to_pinhole(
        self,
        img,
        pinhole_mdl,
        r,
        t,
        r_ph,
        t_ph,
        plane_z_ph=-18.5,
        recompute=False,
        save_maps=False,
    ):

        if not recompute and self.map1 is not None:
            img_undist = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
            return img_undist

        pixel_coord_ph = (
            np.mgrid[0 : pinhole_mdl.height, 0 : pinhole_mdl.width]
            .reshape(2, -1)
            .transpose()
        )
        grid_points_ph = pinhole_mdl.cam2world(pixel_coord_ph)
        grid_points_ph /= grid_points_ph[:, 2, None]
        grid_points_ph *= plane_z_ph
        grid_points_fem = r_ph.apply(grid_points_ph - t_ph, inverse=True)

        grid_points_ocam = r.apply(grid_points_fem) + t
        pixel_coord_ocam = self.world2cam(grid_points_ocam)

        pixel_coord_ocam = pixel_coord_ocam.reshape(
            pinhole_mdl.height, pinhole_mdl.width, 2
        ).astype(np.float32)
        self.map1, self.map2 = cv2.convertMaps(
            pixel_coord_ocam[..., 1], pixel_coord_ocam[..., 0], cv2.CV_16SC2
        )

        img_undist = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

        if save_maps:
            np.savetxt("map_1_0.txt", self.map1[:, :, 0], fmt="%d", delimiter="\n")
            np.savetxt("map_1_1.txt", self.map1[:, :, 1], fmt="%d", delimiter="\n")
            np.savetxt("map_2.txt", self.map2, fmt="%d", delimiter="\n")

        return img_undist

    def undistort(self, img, find_k=False):
        """Undistort an image using the camera model's intrinsic parameters.

        Parameters
        img : array_like, shape (height, width, channels)
            The image to be undistorted in matrix form. The height and width of
            the image must correspond to the height and width of the images
            used for calibration.
        Returns
        -------
        img_undist : numpy.ndarray, shape (height, width, channels)
            The undistorted image. Pixels in the returned image, for which
            there is no correspondent pixels in the given source image,
            are filled with zeros.

        """

        # Create grid using the edge vectors
        grid_points = np.empty((int(self.height) * int(self.width), 3))
        x = np.linspace(
            min(self.edges[0, 0], self.edges[1, 0]),
            max(self.edges[2, 0], self.edges[3, 0]),
            num=int(self.height),
        )
        y = np.linspace(
            min(self.edges[0, 1], self.edges[2, 1]),
            max(self.edges[1, 1], self.edges[3, 1]),
            num=int(self.width),
        )
        x, y = np.meshgrid(x, y, indexing="ij")
        grid_points[:, 0], grid_points[:, 1] = x.reshape(-1), y.reshape(-1)
        grid_points[:, 2] = -np.ones((int(self.height) * int(self.width)))

        # Get pixel coordinates for grid points
        pixel_coord = self.world2cam(grid_points)
        pixel_coord = np.rint(pixel_coord).astype(int)
        valid_idx = np.logical_and(  # TODO clean up
            np.logical_and(pixel_coord[:, 0] >= 0, pixel_coord[:, 0] < self.height),
            np.logical_and(pixel_coord[:, 1] >= 0, pixel_coord[:, 1] < self.width),
        )

        if find_k:
            indices = np.where(valid_idx)[0]
            undistorted_coord = -np.ones((len(indices), 2))
            undistorted_coord[:, 0] = indices // int(self.width)
            undistorted_coord[:, 1] = indices % int(self.width)

            # grid_points[:,2] *= -1
            self.K = self.find_k_intrinsic(
                undistorted_coord, grid_points[valid_idx], N=5
            )

        pixel_coord = pixel_coord[valid_idx]

        if img is not None:
            img = np.asarray(img)
            channels = img.shape[2]
            # Create undistorted image
            img_undist = np.zeros(
                (int(self.height) * int(self.width), channels), dtype=img.dtype
            )
            img_undist[valid_idx] = img[pixel_coord[:, 0], pixel_coord[:, 1], :]

            # Reshape into image
            img_undist = img_undist.reshape(int(self.height), int(self.width), channels)
        else:
            img_undist = None

        if find_k:
            return img_undist, self.K
        else:
            return img_undist

    def find_k_intrinsic(self, pixel_coord, world_coord, N):

        X = np.zeros((2 * N, 5))
        y = np.zeros(2 * N)

        for i in range(N):
            xw, yw, zw = world_coord[i]
            X[2 * i : 2 * i + 2] = np.array(
                [[xw / zw, 0, yw / zw, 1, 0], [0, yw / zw, 0, 0, 1]]
            )
            y[2 * i : 2 * i + 2] = pixel_coord[i]

        opt = np.linalg.lstsq(X, y)[0]

        K = np.array([[opt[0], opt[2], opt[3]], [0, opt[1], opt[4]], [0, 0, 1]])

        return K

    @staticmethod
    def _minrealpolyroot(coeffs):
        roots = polyroots(coeffs)
        realroots = roots[np.isreal(roots) & (np.real(roots) > 0.0)].real
        if realroots.size == 0:
            return np.nan
        else:
            return realroots.min()
