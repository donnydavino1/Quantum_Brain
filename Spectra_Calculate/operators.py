import qutip as qt
import numpy as np

pi = np.pi

""" 2x2 Matrices """
ix, iy, iz = qt.spin_J_set(1 / 2)
sx, sy, sz = qt.spin_J_set(1 / 2)
id = qt.identity(2)

""" 4x4 Matrices """
IDENTITY = qt.tensor(id, id)

Ix = qt.tensor(ix, id)
Iy = qt.tensor(iy, id)
Iz = qt.tensor(iz, id)

Sx = qt.tensor(id, sx)
Sy = qt.tensor(id, sy)
Sz = qt.tensor(id, sz)

IxSx = qt.tensor(ix, sx)
IxSy = qt.tensor(ix, sy)
IxSz = qt.tensor(ix, sz)

IySx = qt.tensor(iy, sx)
IySy = qt.tensor(iy, sy)
IySz = qt.tensor(iy, sz)

IzSx = qt.tensor(iz, sx)
IzSy = qt.tensor(iz, sy)
IzSz = qt.tensor(iz, sz)


""" Density Matrix States """
RHO_1000 = (Iz + Sz + 2 * IzSz).unit()
RHO_0100 = (Iz - Sz - 2 * IzSz).unit()
RHO_0010 = (-Iz + Sz - 2 * IzSz).unit()
RHO_0001 = (-Iz - Sz + 2 * IzSz).unit()

# States without the trailing tails
CLEAN_1000 = (RHO_1000 + IDENTITY / 6).unit()
CLEAN_0100 = (RHO_0100 + IDENTITY / 6).unit()
CLEAN_0010 = (RHO_0010 + IDENTITY / 6).unit()
CLEAN_0001 = (RHO_0001 + IDENTITY / 6).unit()

""" Operators """
# Rotation Operators


def Rx(theta):
    return (-1j * theta * ix).expm()


def Rx_I(theta):
    return qt.tensor(Rx(theta), id)


def Rx_S(theta):
    return qt.tensor(id, Rx(theta))


def Ry(theta):
    return (-1j * theta * iy).expm()


def Ry_I(theta):
    return qt.tensor(Ry(theta), id)


def Ry_S(theta):
    return qt.tensor(id, Ry(theta))


def Rz(theta):
    return (-1j * theta * iz).expm()


def Rz_I(theta):
    return qt.tensor(Rz(theta), id)


def Rz_S(theta):
    return qt.tensor(id, Rz(theta))


def U_J(t, J):
    diagonal = [np.exp(-1j * pi * J * t / 2), np.exp(1j * pi * J * t / 2), np.exp(1j * pi * J * t / 2),
                np.exp(-1j * pi * J * t / 2)]
    return qt.Qobj(np.diag(diagonal), dims=[[2, 2], [2, 2]])


# J coupling with wait time t = 1/(2J). Levitt pg 398
UJ = qt.Qobj((-1j * np.pi * IzSz)).expm()

CNOT = Rx_S(np.pi / 2) * UJ * Ry_S(np.pi / 2)
# cnot_donny = Rx_S(pi / 2) * UJ * Ry_S(pi / 2)



""" Product Operators for Tomography """
# Construct the product operators that correspond to each spectrum. 
# First element operator corresponds to the sum of the peaks and the second element to the difference
# From table 4.1 in Jon Vandermause's thesis, and Leskowitz's paper
ii_real_ops_1 = [Ix, 2 * IxSz]
ii_imag_ops_1 = [-Iy, -2 * IySz]

x1_real_ops_1 = [Ix, 2 * IxSz]
x1_imag_ops_1 = [Iz, 2 * IzSz]

y1_real_ops_1 = [Iz, 2 * IzSz]
y1_imag_ops_1 = [-Iy, -2 * IySz]

x2_real_ops_1 = [Ix, 2 * IxSy]
x2_imag_ops_1 = [-Iy, -2 * IySy]

y2_real_ops_1 = [Ix, -2 * IxSx]
y2_imag_ops_1 = [-Iy, 2 * IySx]

x1x2_real_ops_1 = [Ix, 2 * IxSy]
x1x2_imag_ops_1 = [Iz, 2 * IzSy]

y1y2_real_ops_1 = [Iz, -2 * IzSx]
y1y2_imag_ops_1 = [-Iy, 2 * IySx]

# y1y2j_real_ops_1 = [Iz, Sy]
# y1y2j_imag_ops_1 = [-2*IxSz, 2*IySx]

# y1x2j_real_ops_1 = [Iz, Sx]
# y1x2j_imag_ops_1 = [-2*IxSz, -2*IySy]

# y1y2jx2_real_ops_1 = [Iz, -Sz]
# y1y2jx2_imag_ops_1 = [-2*IxSy, 2*IySx]

product_operators_1 = [ii_real_ops_1, ii_imag_ops_1,
                       x1_real_ops_1, x1_imag_ops_1,
                       y1_real_ops_1, y1_imag_ops_1,
                       x2_real_ops_1, x2_imag_ops_1,
                       y2_real_ops_1, y2_imag_ops_1,
                       x1x2_real_ops_1, x1x2_imag_ops_1,
                       y1y2_real_ops_1, y1y2_imag_ops_1]

ii_real_ops_2 = [Sx, 2 * IzSx]
ii_imag_ops_2 = [-Sy, -2 * IzSy]

x1_real_ops_2 = [Sx, 2 * IySx]
x1_imag_ops_2 = [-Sy, -2 * IySy]

y1_real_ops_2 = [Sx, -2 * IxSx]
y1_imag_ops_2 = [-Sy, 2 * IxSy]

x2_real_ops_2 = [Sx, 2 * IzSx]
x2_imag_ops_2 = [Sz, 2 * IzSz]

y2_real_ops_2 = [Sz, 2 * IzSz]
y2_imag_ops_2 = [-Sy, -2 * IzSy]

x1x2_real_ops_2 = [Sx, 2 * IySx]
x1x2_imag_ops_2 = [Sz, 2 * IySz]

y1y2_real_ops_2 = [Sz, -2 * IxSz]
y1y2_imag_ops_2 = [-Sy, 2 * IxSy]

product_operators_2 = [ii_real_ops_2, ii_imag_ops_2,
                       x1_real_ops_2, x1_imag_ops_2,
                       y1_real_ops_2, y1_imag_ops_2,
                       x2_real_ops_2, x2_imag_ops_2,
                       y2_real_ops_2, y2_imag_ops_2,
                       x1x2_real_ops_2, x1x2_imag_ops_2,
                       y1y2_real_ops_2, y1y2_imag_ops_2]

product_operators = product_operators_1 + product_operators_2


def already_exists(op_new, lst):
    for op_old in lst:
        if np.array_equal(op_new, op_old):
            return True
    return False


def check_basis_complete():
    assert (len(product_operators) == 28)
    all_ops = np.reshape(np.array(product_operators), (len(product_operators) * 2, 4, 4))
    unique_ops = []
    for op_new in all_ops:
        if already_exists(op_new, unique_ops) or already_exists(-op_new, unique_ops):
            continue
        unique_ops.append(op_new)

    if len(unique_ops) == 15:
        print('Operator basis is complete (15 elements)')
    else:
        print(f'something went wrong: we have {len(unique_ops)} unique operators')
