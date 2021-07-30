import pystan
import numpy as np
import pickle
import subprocess
from DavidsNM import save_img
from astropy.io import fits
import matplotlib.pyplot as plt
from copy import deepcopy
import sys

def initfn():
    init_dict = dict(MB = -19 + np.random.random(size = stan_data["n_sn_set"])*0.1,
                     sigma_int = 0.05 + 0.1*np.random.random(size = stan_data["n_sn_set"]),
                     true_x1cs = np.random.random(size = [stan_data["n_sne"], 2])*0.02 + stan_data["obs_mBx1c"][:,1:],
                     x1c_star = np.random.random(size = [stan_data["n_sn_set"], 2])*0.1,
                     true_mB = np.random.random(size = stan_data["n_sne"])*0.02 + stan_data["obs_mBx1c"][:,0],
                     log10_R_x1c = np.random.random(size = [stan_data["n_sn_set"], 2])*0.1,
                     outl_frac = [0.02],
                     A = 0.1,
                     x1c_kern_lengths = [0.2, 0.2]
                )
    print(init_dict)
    return init_dict


stan_data = pickle.load(open(sys.argv[1], 'rb'))

print("n_sne", stan_data["n_sne"])


#del stan_data["names"]
stan_data["n_sn_set"] = len(set(stan_data["sn_set_inds"]))

if min(stan_data["sn_set_inds"]) == 1:
    stan_data["sn_set_inds"] -= 1

for key in stan_data:
    print("stan_data ", key)
    try:
        print(stan_data[key].shape)
    except:
        print(None)


plt.plot(stan_data["z_helio"], stan_data["obs_mBx1c"][:, 0], '.')

plt.savefig("obs_vs_z.pdf")
plt.close()


stan_data["obs_mBx1c_cov"] = np.array([list(item) for item in stan_data["obs_mBx1c_cov"]], dtype=np.float64)
print(stan_data["obs_mBx1c_cov"])

stan_data["allow_alpha_S_N"] = 1
print("SKEW NORMAL!!!!"*100)

pfl_name = subprocess.getoutput("hostname") + "_1DGP.pickle"

f = open("../../stan_code_1DGP.txt", 'r')
lines = f.read()
f.close()

try:
    [sm, sc] = pickle.load(open(pfl_name, 'rb'))
    
    if sc != lines:
        raise_time
except:
    sm = pystan.StanModel(file = "../../stan_code_1DGP.txt")
    pickle.dump([sm, lines], open(pfl_name, 'wb'))
    
fit = sm.sampling(data=stan_data, iter=1000, chains=4, refresh = 10, init = initfn)

print(fit)

fit_params = fit.extract(permuted = True)

pickle.dump((stan_data, fit_params), open("results_" + sys.argv[1], 'wb'))
