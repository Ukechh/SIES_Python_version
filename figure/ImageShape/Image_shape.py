from figure.C2Boundary.C2Boundary import C2Bound
import numpy as np
from scipy.sparse import csr_matrix
import cv2
from figure.ImageShape.boundarydet import boundarydet

class ImgShape(C2Bound):

    def __init__(self, fname, npts, a=1, b=1, dspl=5):
        im = ImgShape.load_img(fname)

        points0, theta0 = boundarydet(0.5, im)

        if dspl >= 1:
            points, tvec, avec, normal = C2Bound.rescale(points0, nbPoints=npts, theta0=theta0, dspl=dspl, nsize=[a,b])
        else:
            points, tvec, avec, normal = C2Bound.rescale_diff(points0, nbPoints=npts, theta0=theta0, nsize=[a,b])
        # Extract name string from filename
        idxslash = fname.rindex('/')
        idxdot = fname.rindex('.')
        nstr = fname[idxslash+1:idxdot]

        # Center of mass not used in this case
        com = None
        
        super().__init__(points, tvec, avec, normal, com, nstr, npts)
    @staticmethod
    def load_img(filename):
        """
        Loads and binarizes an image. Returns a sparse binary matrix indicating boundary position.
        Maximum image size is 500x500.
        """

        img = cv2.imread(filename)
        if img is not None:
            print('Successful extraction!')
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.flipud(img.astype(float) / 255.0)

        if max(img.shape) > 500:
            raise ValueError('Size error: maximum size of the image is 500x500')

        img = np.round((img - np.max(img)) / np.min(img - np.max(img)))

        part_rows, part_cols = np.where(img == 1)
        not_part_rows, not_part_cols = np.where(img == 0)

        sparse_img = csr_matrix((np.ones_like(not_part_rows), 
                               (not_part_rows, not_part_cols)))
        
        trimmed = sparse_img[min(part_rows):max(part_rows)+1, 
                            min(part_cols):max(part_cols)+1]
        
        padded = np.ones((trimmed.shape[0] + 6, trimmed.shape[1] + 6))
        padded[3:-3, 3:-3] = trimmed.toarray()
        
        return padded