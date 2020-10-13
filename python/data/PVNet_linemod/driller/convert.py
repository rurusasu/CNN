from my_pyfun import read_pc_from_ply
import numpy as np
import sys

filename = sys.argv[1]
pc = read_pc_from_ply(filename)
filename = filename.replace(".ply", ".txt")
np.savetxt(filename, pc)

