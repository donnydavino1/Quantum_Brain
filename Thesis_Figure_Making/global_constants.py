import tomography.operators as op
import numpy as np
import qutip as qt

# Matplotlib Resolutions
DPI_SAVE = 800
DPI_DISPLAY = 100

# RGB hexadecimal values
Q1_L_COLOR = "#479FC1"
Q1_R_COLOR = "#0C6178"
Q2_L_COLOR = "#DD935A"
Q2_R_COLOR = "#815509"

# Old ones:
# Q1_L_COLOR = "#C57C56"
# Q1_R_COLOR = "#7B261E"
# Q2_L_COLOR = "#334C80"
# Q2_R_COLOR = "#759EBC"

SPECTRA_NAMES = ['11 real', '11 imag', 'X1 real', 'X1 imag', 'Y1 real', 'Y1 imag',
                 'X2 real', 'X2 imag', 'Y2 real', 'Y2 imag', 'X1X2 real', 'X1X2 imag',
                 'Y1Y2 real', 'Y1Y2 imag']