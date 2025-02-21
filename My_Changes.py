operator.py

# Donny's normalized by hands versions of the above
RHO_1000_normalized = (1/2*Iz + 1/2*Sz + 1*IzSz)
CLEAN_1000_normalized = (1/2*Iz + 1/2*Sz + 1*IzSz+1/4*IDENTITY)
Thermal_normalized = (1/2*Iz + 1/2*Sz+1/4*IDENTITY)



CNOT_Unphased = Rx_S(np.pi / 2) * UJ * Ry_S(np.pi / 2)