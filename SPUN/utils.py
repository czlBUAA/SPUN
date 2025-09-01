from __future__ import print_function
import copy
import dill
import hashlib
import itertools
import bingham_distribution as ms
import math
import numpy as np
import os
import scipy
import scipy.integrate as integrate
import scipy.special
import sys
import torch
import random
import skimage.transform
import caculatation_nc
import logging
import cv2
import se3lib
from PIL import Image
logger = logging.getLogger(__name__)

def generate_coordinates(coords):
    """
    A function that returns all  possible triples of coords

    Parameters:
    coords: a numpy array of coordinates

    Returns:
    x: the first  coordinate of possible triples
    y: the second coordinate of possible triples
    z  the third coordinate of possible triples
    """
    x = coords.reshape(-1, 1).repeat(1, len(coords) * len(coords)).flatten()
    y = coords.reshape(-1, 1).repeat(1, len(coords)).flatten().repeat(len(coords))
    z = coords.reshape(-1, 1).flatten().repeat(len(coords)*len(coords))

    return x, y, z

def load_lookup_table(path):
    """
    Loads lookup table from dill serialized file.

    Returns a table specific tuple. For the Bingham case, the tuple containins:
        table_type (str):
        options (dict): The options used to generate the lookup table.
        res_tensor (numpy.ndarray): The actual lookup table data.
        coords (numpy.ndarray): Coordinates at which lookup table was evaluated.

    For the von Mises case, it contains:
        options (dict): The options used to generate the lookup table.
        res_tensor (numpy.ndarray): The actual lookup table data.
    """
    assert os.path.exists(path), "Lookup table file not found."
    with open(path, "rb") as dillfile:
        return dill.load(dillfile)
    
def build_bd_lookup_table(table_type, options, path=None):
    """
    Builds a lookup table for interpolating the bingham normalization
    constant.  If a lookup table with the given options already exists, it is
    loaded and returned instead of building a new one.

    Arguments:
        table_type: Type of lookup table used. May be 'uniform' or 'nonuniform'
        options: Dict cotaining type specific options.
            If type is "uniform" this dict must contain:
                "bounds" = Tuple (lower_bound, upper_bound) representing bounds.
                "num_points" = Number of points per dimension.
            If type is "nonuniform" this dict must contain a key "coords" which
            is a numpy arrays representing the coordinates at which the
            interpolation is evaluated.
        path: absolute path for the lookup table (optional). The default is to
            create a hash based on the options and to use this for constructing
            a file name and placing the file in the precomputed folder.
    """
    hash_obj = hashlib.sha256()
    hash_obj.update(table_type)
    hash_obj.update(dill.dumps(options))
    config_hash = hash_obj.hexdigest()

    if not path:
        path = os.path.dirname(__file__) \
               + "/../precomputed/lookup_{}.dill".format(config_hash)

    # Load existing table or create new one.
    if os.path.exists(path):
        with open(path, "rb") as dillfile:
            (serialized_type, serialized_options, res_table, coords) \
                = dill.load(dillfile)
            hash_obj = hashlib.sha256()
            hash_obj.update(serialized_type)
            hash_obj.update(dill.dumps(serialized_options))
            file_config_hash = hash_obj.hexdigest()
            assert file_config_hash == config_hash, \
                "Serialized lookup table does not match given type & options."

    elif table_type == "uniform":
        # Number of points per axis.
        (lbound, rbound) = options["bounds"]
        num_points = options["num_points"]

        assert num_points > 1, \
            "Grid must have more than one point per dimension."

        nc_options = {"epsrel": 1e-3, "epsabs": 1e-7}

        coords = np.linspace(lbound, rbound, num_points)

        res_table = _compute_bd_lookup_table(coords, nc_options)

        with open(path, "wb") as dillfile:
            dill.dump((table_type, options, res_table, coords), dillfile)

    elif table_type == "nonuniform":
        nc_options = {"epsrel": 1e-3, "epsabs": 1e-7}

        coords = options["coords"]

        res_table = _compute_bd_lookup_table(coords, nc_options)

        with open(path, "wb") as dillfile:
            dill.dump((table_type, options, res_table, coords), dillfile)

    else:
        sys.exit("Unknown lookup table type")

    return res_table
def _compute_bd_lookup_table(coords, nc_options):
    num_points = len(coords)

    pool = Pool()

    def nc_wrapper(idx):
        pt_idx = point_indices[idx]

        # Indexing pt_idx in the order 2,1,0 vs. 0,1,2 has no impact
        # on the result as the Bingham normalization constant is agnostic to it.
        # However, the numpy integration that is used to compute it, combines
        # numerical 2d and 1d integration which is why the order matters for the
        # actual computation time.
        #
        # TODO: Make pymanstats choose best order automatically.
        norm_const = ms.BinghamDistribution.normalization_constant(
            np.array(
                [coords[pt_idx[2]], coords[pt_idx[1]], coords[pt_idx[0]], 0.]),
            "numerical", nc_options)
        print("Computing NC for Z=[{}, {}, {}, 0.0]: {}".format(
            coords[pt_idx[2]], coords[pt_idx[1]], coords[pt_idx[0]],
            norm_const))
        return norm_const

    point_indices = list(itertools.combinations_with_replacement(
        range(0, num_points), 3))
    results = pool.map(nc_wrapper, range(len(point_indices)))
   

    res_tensor = -np.ones((num_points, num_points, num_points))
    for idx_pos, pt_idx in enumerate(point_indices):
        res_tensor[pt_idx[0], pt_idx[1], pt_idx[2]] = results[idx_pos]
        res_tensor[pt_idx[0], pt_idx[2], pt_idx[1]] = results[idx_pos]
        res_tensor[pt_idx[1], pt_idx[0], pt_idx[2]] = results[idx_pos]
        res_tensor[pt_idx[1], pt_idx[2], pt_idx[0]] = results[idx_pos]
        res_tensor[pt_idx[2], pt_idx[0], pt_idx[1]] = results[idx_pos]
        res_tensor[pt_idx[2], pt_idx[1], pt_idx[0]] = results[idx_pos]

    return res_tensor

class AverageMeter(object):
    """Computes and stores the averages over a numbers or dicts of numbers.

    For the dict, this class assumes that no new keys are added during
    the computation.
    """

    def __init__(self):
        self.last_val = 0
        self.avg = 0 
        self.count = 0 

    def update(self, val, n=1):
        self.last_val = val
        n = float(n)
        if type(val) == dict:
            if self.count == 0:
                self.avg = copy.deepcopy(val)
            else:
                for key in val:
                    self.avg[key] *= self.count / (self.count + n)
                    self.avg[key] += val[key] * n / (self.count + n)
        else:
            self.avg *= self.count / (self.count + n)
            self.avg += val * n / (self.count + n)

        self.count += n
        self.last_val = val

def normalization_constant(param_z, mode="default", options=dict()):
        """Computes the Bingham normalization constant.

        Parameters
        ----------
        param_z : array of shape (dim)
            Diagonal entries of dispersion parameter matrix Z of the Bingham
            distribution.
        mode : string
            Method of computation (optional).
        options : dict
            Computation-method specific options.
        """
        # Gerhard Kurz, Igor Gilitschenski, Simon Julier, Uwe D. Hanebeck,
        # "Recursive Bingham Filter for Directional Estimation Involving 180
        # Degree Symmetry", Journal of Advances in Information
        # Fusion, 9(2):90 - 105, December 2014.

        bd_dim = param_z.shape[0]

        assert bd_dim in BinghamDistribution.IMPLEMENTED_DIMENSIONS \
            and param_z.ndim == 1, \
            "param_z needs to be a vector of supported dimension."

        # TODO Check structure of Z

        if bd_dim == 2:
            if mode == "default" or mode == "bessel":
                # Surface area of the unit sphere is a factor in the
                # normalization constant. The formula is taken from
                # https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area
                sphere_surface_area = 2.0 * (np.pi**(bd_dim / 2.0) /
                                             scipy.special.gamma(bd_dim / 2.0))

                norm_const = (np.exp(param_z[1]) * sphere_surface_area *
                              scipy.special.iv(
                                  0, (param_z[0] - param_z[1]) / 2.0)
                              * np.exp((param_z[0] - param_z[1]) / 2.0))
                return norm_const
        elif bd_dim == 4:
            if mode == "default" or mode == "saddlepoint":
                f = BinghamDistribution.__norm_const_saddlepoint(
                    np.sort(-param_z)+1)
                f *= np.exp(1)
                return f[2]
            elif mode == "numerical":
                param_z_diag = np.diag(param_z)

                def bd_likelihood(x):
                    return np.exp(np.dot(x, np.dot(param_z_diag, x)))

                def integrand(phi1, phi2, phi3):
                    sp1 = np.sin(phi1)
                    sp2 = np.sin(phi2)
                    return bd_likelihood(np.array([
                        sp1 * sp2 * np.sin(phi3),
                        sp1 * sp2 * np.cos(phi3),
                        sp1 * np.cos(phi2),
                        np.cos(phi1)
                    ])) * (sp1 ** 2.) * sp2

                norm_const = integrate.tplquad(
                    integrand,
                    0.0, 2.0 * np.pi,  # phi3
                    lambda x: 0.0, lambda x: np.pi,  # phi2
                    lambda x, y: 0.0, lambda x, y: np.pi,  # phi1
                    **options
                ) 

                return norm_const[0]

        sys.exit("Invalid computation mode / dimension combination.")
def skew(X):
    return torch.tensor([[0, -X[2], X[1]],
                         [X[2], 0, -X[0]],
                         [-X[1], X[0], 0]], device=X.device, dtype=X.dtype)

def Qmultiply(X, Y):
    skew_X = skew(X[1:])
    term1 = torch.matmul(skew_X, Y[1:]) + X[0] * Y[1:] + Y[0] * X[1:]
    term2 = X[0] * Y[0] - torch.dot(X[1:], Y[1:])
    term2 = torch.tensor(term2.item(), device=X.device, dtype=X.dtype).unsqueeze(0)
    return torch.cat((torch.tensor(term2, device=X.device, dtype=X.dtype), term1), dim=0)

def expand_qt(qt):
    N = qt.shape[0]
    if qt.is_cuda:
            device = qt.get_device()
    else:
        device = "cpu"
    expanded_qt = torch.cat((torch.zeros(N, 1,  device=device, dtype=qt.dtype), qt), dim=1)
    #scalar first
    return expanded_qt

def q2R(q,  order="speed"):
    if q.shape != (4,):
        raise ValueError("Input quaternion must have shape (4,)")
    
    
    q0 = q[0]
    q_vec = q[1:]
    
    # Top row: [q0, -q_vec^T]
    top_row = torch.cat((q0.unsqueeze(0), -q_vec), dim=0).unsqueeze(0)  # shape: (1, 4)
    
    # Bottom block:
    # First column: q_vec (as a column vector)
    q_vec_col = q_vec.unsqueeze(1)  # shape: (3, 1)
    # Next three columns: -skew(q_vec) + q0 * I3
    I3 = torch.eye(3, dtype=q.dtype, device=q.device)
    q_vec_skew = skew(q_vec)
    bottom_right = -q_vec_skew + q0 * I3  # shape: (3, 3)
    bottom = torch.cat((q_vec_col, bottom_right), dim=1)  # shape: (3, 4)
    
    # Concatenate top row and bottom block to form the final 4x4 matrix
    Rqp = torch.cat((top_row, bottom), dim=0)
    return Rqp

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        scale = min_dim / min(h, w)
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode != "crop":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image using bilinear interpolation
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(h * scale), round(w * scale)),
            order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        if len(image.shape)>2:
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        else:
            padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0

        if len(image.shape) > 2:
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        else:
            padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop
def eaad_bingham(bingham_z, integral_options=None):
    """ Expected Absolute Angular Deviation of Bingham Random Vector

    Arguments:
        bingham_z: Bingham dispersion parameter in the format expected by the
            manstats BinghamDistribution class.
        integral_options: Options to pass on to the scipy integrator for
            computing the eaad and the bingham normalization constant.
    """

    def aad(quat_a, quat_b):
        # acos_val = np.arccos(np.dot(quat_a, quat_b))
        # diff_ang = 2 * np.min([acos_val, np.pi - acos_val])
        acos_val = np.arccos(np.abs(np.dot(quat_a, quat_b)))
        diff_ang = 2 * acos_val
        return diff_ang**2

    if integral_options is None:
        integral_options = {"epsrel": 1e-4, "epsabs": 1e-4}
    
    mode = np.array([1,0,0,0])
    def integrand_transformed(x):
        # To avoid unnecessary divisions, this term does not contain the
        # normalization constant. At the end, the result of the integration is
        # divided by it.
        return aad(x, mode) \
               * np.exp(np.dot(x, np.dot(np.diag(bingham_z), x)))

    def integrand(phi1, phi2, phi3):
        sp1 = np.sin(phi1)
        sp2 = np.sin(phi2)
        return integrand_transformed(np.array([
            sp1 * sp2 * np.sin(phi3),
            sp1 * sp2 * np.cos(phi3),
            sp1 * np.cos(phi2),
            np.cos(phi1)
        ])) * (sp1 ** 2.) * sp2

    eaad_int = integrate.tplquad(
        integrand,
        0.0, 2.0 * np.pi,  # phi3
        lambda x: 0.0, lambda x: np.pi,  # phi2
        lambda x, y: 0.0, lambda x, y: np.pi,  # phi1
        **integral_options
    )
    caculatation_nc.t0 = - np.max(bingham_z)-2
    norm_const = caculatation_nc.nc(bingham_z)
    return np.sqrt(eaad_int[0] / norm_const)

def eatd_Guassian(cov):
   
    result = np.trace(cov)
    
    return np.sqrt(result)
def vec_to_bingham_z_many(y):
    z = -torch.exp(y).cumsum(1)[:, [2, 1, 0]].unsqueeze(0)
    return z 


def vec_to_bingham_z(y):
    z = -torch.exp(y).cumsum(0)[[2, 1, 0]].unsqueeze(0)
    if not all(z[0][:-1] <= z[0][1:]):
        print(z)
    return z

def rotate_cam(image, t, q, K, magnitude):
    """ Apply warping corresponding to a random camera rotation
    Arguments:
     - image: Input image
     - t, q: Object pose (location,orientation)
     - K: Camera calibration matrix
     - magnitude: 2 * maximum perturbation per angle in deg
    Return:
        - image_warped: Output image
        - t_new, q_new: Updated object pose
    """
    image = np.array(image, dtype=np.uint8)
    pyr_change = (np.random.rand(3)-0.5)*magnitude

    R_change = se3lib.euler2SO3_left(pyr_change[0], pyr_change[1], pyr_change[2])

    # Construct warping (perspective) matrix
    M = K*R_change*np.linalg.inv(K)

    height, width = np.shape(image)[:2]
    image_warped = cv2.warpPerspective(image, M, (width, height), cv2.WARP_INVERSE_MAP)

    # Update pose
    t_new = np.array(np.matrix(t)*R_change.T)[0]
    q_change = se3lib.SO32quat(R_change)
    q_new = np.array(se3lib.quat_mult(q_change, q))[0]
     # 将 BGR 转换为 RGB
    image_rgb = cv2.cvtColor(image_warped, cv2.COLOR_BGR2RGB)
    # 转换为 PIL 图像
    pil_image = Image.fromarray(image_rgb)
    
    return pil_image, t_new, q_new

def rotate_image(image, t, q, K):
    """ Apply warping corresponding to a random in-plane rotation
     Arguments:
      - image: Input image
      - t, q: Object pose (location,orientation)
      - K: Camera calibration matrix
      - magnitude: 2 * maximum perturbation per roll
     Return:
         - image_warped: Output image
         - t_new, q_new: Updated object pose
    """
    image = np.array(image, dtype=np.uint8)
    change = (np.random.rand(1)-0.5)*170

    R_change = se3lib.euler2SO3_left(0, 0, change[0])

    # Construct warping (perspective) matrix
    M = K*R_change*np.linalg.inv(K)

    height, width = np.shape(image)[:2]
    image_warped = cv2.warpPerspective(image, M, (width, height), cv2.WARP_INVERSE_MAP)
    image_rgb = cv2.cvtColor(image_warped, cv2.COLOR_BGR2RGB)
    # 转换为 PIL 图像
    pil_image = Image.fromarray(image_rgb)
    # Update pose
    t_new = np.array(np.matrix(t)*R_change.T)[0]
    q_change = se3lib.SO32quat(R_change)
    q_new = np.array(se3lib.quat_mult(q_change, q))[0]

    return pil_image, t_new, q_new
