import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.insert(1, os.path.dirname(os.getcwd()))
from spaces import sspspace


# Try changing ssp_type (hex or rand), ssp_dim or n_scales/n_rotates, length_scale, bounds
# Note: similarity_plot only works for domain_dim=1 or 2 

# Create SSP space
domain_dim = 2 # The dim of the 'x' space
bounds = np.tile([-1,1],(domain_dim,1)) # bounds of x space (needed only for decoding, can set as None if you don't need to decode)
ssp_type = 'rand'
if ssp_type=='hex':
    ssp_space = sspspace.HexagonalSSPSpace(domain_dim,
                     n_scales=8,n_rotates=5, # You can change the dim of the SSP either via the ssp_dim arg or (in the case of hex ssps) n_scales and n_rotates. try changing these to see what happens
                     domain_bounds=bounds, length_scale=0.1) # I use lenth_scale=0.1 for bounds [-1,1] and =1 for [-10,10], as a rough starting point
elif ssp_type=='rand':
    ssp_space = sspspace.RandomSSPSpace(domain_dim,
                     ssp_dim=151, domain_bounds=bounds, length_scale=0.1)

# For HexSSPs, only certain dims are allowed. If you make the space with an invalid ssp_dim arg, it will just round ssp_dim to the closest 'ok' one, so you might need to check the ssp_dim of the returned ssp_space
d = ssp_space.ssp_dim 

# Some random x
x = np.array([0.1,-0.4])
S = ssp_space.encode(x)

plt.figure()
ax = plt.subplot(111)
im = ssp_space.similarity_plot(S, ax=ax)
plt.colorbar(im)
plt.show()

# Let's try decoding
xhat = ssp_space.decode(S, method='from-set') 
print(x,S,xhat)

print(np.sqrt(np.mean((x-xhat)**2)))


# methods: from-set, direct-optim, network (requries tensorflow, sklearn and for you to run ssp_space.train_decoder_net)
