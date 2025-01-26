import qutip as qt
from qutip import Qobj
import numpy as np
from dataclasses import dataclass, field
from typing import ClassVar

pi = np.pi

def gradient(rho: Qobj) -> Qobj:
    return qt.qdiags(rho.diag(), offsets=0, dims=rho.dims, shape=rho.shape)

@dataclass(frozen=True)
class ThreeSpinOperators:
    """
    A container for 3-spin product operators in QuTiP, plus
    some utility methods for rotations and J-coupling evolution.
    """
    j12: float
    j13: float
    j23: float

    # -- 2x2 single-spin operators (ClassVars, shared across instances) --
    ix: ClassVar[Qobj] = qt.spin_J_set(1/2)[0]
    iy: ClassVar[Qobj] = qt.spin_J_set(1/2)[1]
    iz: ClassVar[Qobj] = qt.spin_J_set(1/2)[2]
    one: ClassVar[Qobj] = qt.identity(2)

    # -- 8x8 identity --
    identity: ClassVar[Qobj] = qt.tensor(one, one, one)

    # -- Single Operators --
    I1x: ClassVar[Qobj] = qt.tensor(ix,  one, one)
    I1y: ClassVar[Qobj] = qt.tensor(iy,  one, one)
    I1z: ClassVar[Qobj] = qt.tensor(iz,  one, one)

    I2x: ClassVar[Qobj] = qt.tensor(one, ix,  one)
    I2y: ClassVar[Qobj] = qt.tensor(one, iy,  one)
    I2z: ClassVar[Qobj] = qt.tensor(one, iz,  one)

    I3x: ClassVar[Qobj] = qt.tensor(one, one, ix)
    I3y: ClassVar[Qobj] = qt.tensor(one, one, iy)
    I3z: ClassVar[Qobj] = qt.tensor(one, one, iz)

    # -- Double Operators: I1 x I2 --
    I1xI2x: ClassVar[Qobj] = qt.tensor(ix, ix, one)
    I1yI2x: ClassVar[Qobj] = qt.tensor(iy, ix, one)
    I1zI2x: ClassVar[Qobj] = qt.tensor(iz, ix, one)

    I1xI2y: ClassVar[Qobj] = qt.tensor(ix, iy, one)
    I1yI2y: ClassVar[Qobj] = qt.tensor(iy, iy, one)
    I1zI2y: ClassVar[Qobj] = qt.tensor(iz, iy, one)

    I1xI2z: ClassVar[Qobj] = qt.tensor(ix, iz, one)
    I1yI2z: ClassVar[Qobj] = qt.tensor(iy, iz, one)
    I1zI2z: ClassVar[Qobj] = qt.tensor(iz, iz, one)

    # -- I1 x I3 --
    I1xI3x: ClassVar[Qobj] = qt.tensor(ix, one, ix)
    I1yI3x: ClassVar[Qobj] = qt.tensor(iy, one, ix)
    I1zI3x: ClassVar[Qobj] = qt.tensor(iz, one, ix)

    I1xI3y: ClassVar[Qobj] = qt.tensor(ix, one, iy)
    I1yI3y: ClassVar[Qobj] = qt.tensor(iy, one, iy)
    I1zI3y: ClassVar[Qobj] = qt.tensor(iz, one, iy)

    I1xI3z: ClassVar[Qobj] = qt.tensor(ix, one, iz)
    I1yI3z: ClassVar[Qobj] = qt.tensor(iy, one, iz)
    I1zI3z: ClassVar[Qobj] = qt.tensor(iz, one, iz)

    # -- I2 x I3 --
    I2xI3x: ClassVar[Qobj] = qt.tensor(one, ix, ix)
    I2yI3x: ClassVar[Qobj] = qt.tensor(one, iy, ix)
    I2zI3x: ClassVar[Qobj] = qt.tensor(one, iz, ix)

    I2xI3y: ClassVar[Qobj] = qt.tensor(one, ix, iy)
    I2yI3y: ClassVar[Qobj] = qt.tensor(one, iy, iy)
    I2zI3y: ClassVar[Qobj] = qt.tensor(one, iz, iy)

    I2xI3z: ClassVar[Qobj] = qt.tensor(one, ix, iz)
    I2yI3z: ClassVar[Qobj] = qt.tensor(one, iy, iz)
    I2zI3z: ClassVar[Qobj] = qt.tensor(one, iz, iz)

    # -- Triple Operators --
    I1xI2xI3x: ClassVar[Qobj] = qt.tensor(ix, ix, ix)
    I1yI2xI3x: ClassVar[Qobj] = qt.tensor(iy, ix, ix)
    I1zI2xI3x: ClassVar[Qobj] = qt.tensor(iz, ix, ix)

    I1xI2yI3x: ClassVar[Qobj] = qt.tensor(ix, iy, ix)
    I1yI2yI3x: ClassVar[Qobj] = qt.tensor(iy, iy, ix)
    I1zI2yI3x: ClassVar[Qobj] = qt.tensor(iz, iy, ix)

    I1xI2zI3x: ClassVar[Qobj] = qt.tensor(ix, iz, ix)
    I1yI2zI3x: ClassVar[Qobj] = qt.tensor(iy, iz, ix)
    I1zI2zI3x: ClassVar[Qobj] = qt.tensor(iz, iz, ix)

    I1xI2xI3y: ClassVar[Qobj] = qt.tensor(ix, ix, iy)
    I1yI2xI3y: ClassVar[Qobj] = qt.tensor(iy, ix, iy)
    I1zI2xI3y: ClassVar[Qobj] = qt.tensor(iz, ix, iy)

    I1xI2yI3y: ClassVar[Qobj] = qt.tensor(ix, iy, iy)
    I1yI2yI3y: ClassVar[Qobj] = qt.tensor(iy, iy, iy)
    I1zI2yI3y: ClassVar[Qobj] = qt.tensor(iz, iy, iy)

    I1xI2zI3y: ClassVar[Qobj] = qt.tensor(ix, iz, iy)
    I1yI2zI3y: ClassVar[Qobj] = qt.tensor(iy, iz, iy)
    I1zI2zI3y: ClassVar[Qobj] = qt.tensor(iz, iz, iy)

    I1xI2xI3z: ClassVar[Qobj] = qt.tensor(ix, ix, iz)
    I1yI2xI3z: ClassVar[Qobj] = qt.tensor(iy, ix, iz)
    I1zI2xI3z: ClassVar[Qobj] = qt.tensor(iz, ix, iz)

    I1xI2yI3z: ClassVar[Qobj] = qt.tensor(ix, iy, iz)
    I1yI2yI3z: ClassVar[Qobj] = qt.tensor(iy, iy, iz)
    I1zI2yI3z: ClassVar[Qobj] = qt.tensor(iz, iy, iz)

    I1xI2zI3z: ClassVar[Qobj] = qt.tensor(ix, iz, iz)
    I1yI2zI3z: ClassVar[Qobj] = qt.tensor(iy, iz, iz)
    I1zI2zI3z: ClassVar[Qobj] = qt.tensor(iz, iz, iz)

    # -------------------------------------------------------------------
    # Rotation Operators (angles in radians!)
    # -------------------------------------------------------------------
    def rot(self, axis: str, angle: float) -> Qobj:
        """
        Single-spin rotation on 2x2 operator (ix, iy, or iz),
        then exponentiate to get the unitary.
        """
        if axis == 'x':
            return (-1j * angle * self.ix).expm()
        elif axis == 'y':
            return (-1j * angle * self.iy).expm()
        elif axis == 'z':
            return (-1j * angle * self.iz).expm()
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

    def rot_1(self, axis: str, angle: float) -> Qobj:
        """Rotation on spin 1 only."""
        return qt.tensor(self.rot(axis, angle), self.one, self.one)

    def rot_2(self, axis: str, angle: float) -> Qobj:
        """Rotation on spin 2 only."""
        return qt.tensor(self.one, self.rot(axis, angle), self.one)

    def rot_3(self, axis: str, angle: float) -> Qobj:
        """Rotation on spin 3 only."""
        return qt.tensor(self.one, self.one, self.rot(axis, angle))

    def UJ(self, t: float) -> Qobj:
        """
        Returns the unitary evolution under scalar couplings j12, j13, j23 for time t.
        H = 2*pi * (j12*I1zI2z + j13*I1zI3z + j23*I2zI3z)
        """
        ham: Qobj = self.j12 * self.I1zI2z + self.j13 * self.I1zI3z + self.j23 * self.I2zI3z
        ham = 2 * pi * ham
        return (-1j * ham * t).expm()
