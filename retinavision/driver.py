from ssnn import SSNN
from rf_generation import rf_ozimek
import utils
import cortex_generation
import os


tesselation = SSNN(N=5000, name="1k_retina", fov=0.2, max_iter=100, V = True)
utils.writePickle('{0}/tesselation_1k.mat'.format(os.getcwd()), tesselation)

r_loc, r_coeff, fov_dist_5 = rf_ozimek(tesselation, sigma_power=1.1, kernel_ratio=3.0, sigma_base=1.0, mean_rf=5)
utils.writePickle('ret1k_loc.pkl', r_loc)
utils.writePickle('ret1k_coeff.pkl', r_coeff)

l_split, r_split = cortex_generation.LRsplit(r_loc)
L_loc, R_loc = cortex_generation.cort_map(l_split, r_split)
L_loc_final, R_loc_final, L_coeff, R_coeff, cort_size = cortex_generation.cort_prepare(L_loc, R_loc)

utils.writePickle('1k_Lloc_tight.pkl', L_loc_final)
utils.writePickle('1k_Rloc_tight.pkl', R_loc_final)

utils.writePickle('1k_Lcoeff_tight.pkl', L_coeff)
utils.writePickle('1k_Rcoeff_tight.pkl', R_coeff)




