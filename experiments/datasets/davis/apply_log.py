import numpy as np

y = np.loadtxt("affinity.txt")
# From https://doi.org/10.1093/bioinformatics/bty593
#   log_y = - np.log10(y / 1e9)
# But we use 4 (the max) to start from 0
log_y = 4 - np.log10(y)
np.savetxt("log_affinity.txt", log_y)

print("Saved to log_affinity.txt")
