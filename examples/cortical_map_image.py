import cv2
import numpy as np
from retinavision.retina import Retina
from retinavision.cortex import Cortex
from retinavision import datadir, utils
from os.path import join
import os

#Create and load retina
R = Retina()
R.info()
R.loadLoc(join(datadir, "retinas", "ret1k_loc.pkl"))
R.loadCoeff(join(datadir, "retinas", "ret1k_coeff.pkl"))

img = cv2.imread("{}/examples/mario_state_test_2.png".format(os.getcwd()), cv2.IMREAD_GRAYSCALE)
print(os.getcwd())
print(type(img))

#Prepare retina
x = img.shape[1]/2
y = img.shape[0]/2
print(x)
print(y)
print(img.shape)
fixation = (y,x)
R.prepare(img.shape, fixation)

#Create and prepare cortex
C = Cortex()
lp = join(datadir, "cortices", "1k_cort_leftloc.pkl")
rp = join(datadir, "cortices", "1k_cort_rightloc.pkl")
C.loadLocs(lp, rp)
C.loadCoeffs(join(datadir, "cortices", "1k_cort_leftcoeff.pkl"), join(datadir, "cortices", "1k_cort_rightcoeff.pkl"))

V = R.sample(img, fixation)
tight = R.backproject_tight_last()
cimg = C.cort_img(V)
cv2.imwrite("{}/examples/mario_cortical_1k.png".format(os.getcwd()), cimg)
cv2.imwrite("{}/examples/mario_backprojection_1k.png".format(os.getcwd()), tight)
# cv2.namedWindow("inverted", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("inverted", tight) 
        
# cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("input", img) 
        
# cv2.namedWindow("cortical", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("cortical", cimg)
        
# key = cv2.waitKey(10)
        
