import qutip as qt
from qutip import Qobj
import numpy as np
from dataclasses import dataclass

pi = np.pi


def gradient(rho: Qobj) -> Qobj:
    return qt.qdiags(rho.diag(), offsets=0, dims=rho.dims, shape=rho.shape)


@dataclass(frozen=True)
class ThreeSpinOperators:
    j12: float
    j13: float
    j23: float

    """ 2x2 matrices """
    ix, iy, iz = qt.spin_J_set(1/2)
    one: Qobj = qt.identity(2)

    """ 8x8 Product Operators (4x4x4 many) """
    identity: Qobj = qt.tensor(one, one, one)

    # Single Operators
    I1x: Qobj = qt.tensor(ix, one, one)
    I1y: Qobj = qt.tensor(iy, one, one)
    I1z: Qobj = qt.tensor(iz, one, one)

    I2x: Qobj = qt.tensor(one, ix, one)
    I2y: Qobj = qt.tensor(one, iy, one)
    I2z: Qobj = qt.tensor(one, iz, one)

    I3x: Qobj = qt.tensor(one, one, ix)
    I3y: Qobj = qt.tensor(one, one, iy)
    I3z: Qobj = qt.tensor(one, one, iz)

    # Double Operators
    # I1 x I2
    I1xI2x: Qobj = qt.tensor(ix, ix, one)
    I1yI2x: Qobj = qt.tensor(iy, ix, one)
    I1zI2x: Qobj = qt.tensor(iz, ix, one)

    I1xI2y: Qobj = qt.tensor(ix, iy, one)
    I1yI2y: Qobj = qt.tensor(iy, iy, one)
    I1zI2y: Qobj = qt.tensor(iz, iy, one)

    I1xI2z: Qobj = qt.tensor(ix, iz, one)
    I1yI2z: Qobj = qt.tensor(iy, iz, one)
    I1zI2z: Qobj = qt.tensor(iz, iz, one)

    # I1 x I3
    I1xI3x: Qobj = qt.tensor(ix, one, ix)
    I1yI3x: Qobj = qt.tensor(iy, one, ix)
    I1zI3x: Qobj = qt.tensor(iz, one, ix)

    I1xI3y: Qobj = qt.tensor(ix, one, iy)
    I1yI3y: Qobj = qt.tensor(iy, one, iy)
    I1zI3y: Qobj = qt.tensor(iz, one, iy)

    I1xI3z: Qobj = qt.tensor(ix, one, iz)
    I1yI3z: Qobj = qt.tensor(iy, one, iz)
    I1zI3z: Qobj = qt.tensor(iz, one, iz)

    # I2 x I3
    I2xI3x: Qobj = qt.tensor(one, ix, ix)
    I2yI3x: Qobj = qt.tensor(one, iy, ix)
    I2zI3x: Qobj = qt.tensor(one, iz, ix)

    I2xI3y: Qobj = qt.tensor(one, ix, iy)
    I2yI3y: Qobj = qt.tensor(one, iy, iy)
    I2zI3y: Qobj = qt.tensor(one, iz, iy)

    I2xI3z: Qobj = qt.tensor(one, ix, iz)
    I2yI3z: Qobj = qt.tensor(one, iy, iz)
    I2zI3z: Qobj = qt.tensor(one, iz, iz)

    # Triple Operators
    I1xI2xI3x: Qobj = qt.tensor(ix, ix, ix)
    I1yI2xI3x: Qobj = qt.tensor(iy, ix, ix)
    I1zI2xI3x: Qobj = qt.tensor(iz, ix, ix)

    I1xI2yI3x: Qobj = qt.tensor(ix, iy, ix)
    I1yI2yI3x: Qobj = qt.tensor(iy, iy, ix)
    I1zI2yI3x: Qobj = qt.tensor(iz, iy, ix)

    I1xI2zI3x: Qobj = qt.tensor(ix, iz, ix)
    I1yI2zI3x: Qobj = qt.tensor(iy, iz, ix)
    I1zI2zI3x: Qobj = qt.tensor(iz, iz, ix)

    I1xI2xI3y: Qobj = qt.tensor(ix, ix, iy)
    I1yI2xI3y: Qobj = qt.tensor(iy, ix, iy)
    I1zI2xI3y: Qobj = qt.tensor(iz, ix, iy)

    I1xI2yI3y: Qobj = qt.tensor(ix, iy, iy)
    I1yI2yI3y: Qobj = qt.tensor(iy, iy, iy)
    I1zI2yI3y: Qobj = qt.tensor(iz, iy, iy)

    I1xI2zI3y: Qobj = qt.tensor(ix, iz, iy)
    I1yI2zI3y: Qobj = qt.tensor(iy, iz, iy)
    I1zI2zI3y: Qobj = qt.tensor(iz, iz, iy)

    I1xI2xI3z: Qobj = qt.tensor(ix, ix, iz)
    I1yI2xI3z: Qobj = qt.tensor(iy, ix, iz)
    I1zI2xI3z: Qobj = qt.tensor(iz, ix, iz)

    I1xI2yI3z: Qobj = qt.tensor(ix, iy, iz)
    I1yI2yI3z: Qobj = qt.tensor(iy, iy, iz)
    I1zI2yI3z: Qobj = qt.tensor(iz, iy, iz)

    I1xI2zI3z: Qobj = qt.tensor(ix, iz, iz)
    I1yI2zI3z: Qobj = qt.tensor(iy, iz, iz)
    I1zI2zI3z: Qobj = qt.tensor(iz, iz, iz)

    """" Rotation Operators (angles in radians!) """
    def rot(self, axis: str, angle: float) -> Qobj:
        if axis == 'x':
            return (-1j * angle * self.ix).expm()
        elif axis == 'y':
            return (-1j * angle * self.iy).expm()
        elif axis == 'z':
            return (-1j * angle * self.iz).expm()
        else:
            raise ValueError("axis has to be 'x', 'y', or 'z'")

    def rot_1(self, axis: str, angle: float) -> Qobj:
        return qt.tensor(self.rot(axis, angle), self.one, self.one)

    def rot_2(self, axis: str, angle: float) -> Qobj:
        return qt.tensor(self.one, self.rot(axis, angle), self.one)

    def rot_3(self, axis: str, angle: float) -> Qobj:
        return qt.tensor(self.one, self.one, self.rot(axis, angle))

    def UJ(self, t: float) -> Qobj:
        ham: Qobj = self.j12 * self.I1zI2z + self.j13 * self.I1zI3z + self.j23 * self.I2zI3z
        ham: Qobj = 2 * pi * ham
        return (-1j * ham * t).expm()
