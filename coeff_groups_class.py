from dataclasses import dataclass, field

import numpy as np
import qutip as qt
import operators as op

"""
"coefficient groups" is a list of "group"s, with each group being a list of:

`[operator, (c1, spectrum type), (c2, spectrum type), (c3, spectrum type), ...]`

where the first element of the group is a product operator,

and the following elements are tuples of: (coefficients corresponding to that operator, 
the type of spectrum which the coefficient came from).

In the case of ADP there are two spectrum types: P1 and P2

(each spectrum produces two coefficients)
"""


@dataclass
class CoefficientGroups:
    data: list[list] = field(default_factory=list)

    def add_coefficient(self, operator: qt.Qobj, c: float, atom: str):
        for group in self.data:
            if np.array_equal(operator, group[0]):
                group.append((c, atom))
                return
            elif np.array_equal(-operator, group[0]):
                group.append((-c, atom))
                return
        self.data.append([operator, (c, atom)])

    def reconstruct_rho(self, abs_trace=True, positive_diag=False) -> qt.Qobj:
        """
        Given a 'coefficient groups', construct a density matrix.
        Could play around setting abs_trace to True/False and seeing which gives
        us better projection & fidelity values.
        """
        assert len(self.data) == 15, "number of groups should be 15!"

        rho = qt.Qobj(np.zeros(self.data[0][0].shape), dims=self.data[0][0].dims)
        for group in self.data:
            coefficients = [c for (c, atom) in group[1:]]
            # TODO: implement error propagation
            rho += group[0] * np.mean(coefficients)
        rho = clean_dm(rho, abs_trace=abs_trace, positive_diag=positive_diag)
        return qt.Qobj(rho)

    def get_error(self, abs_trace=True, positive_diag=False) -> qt.Qobj:
        groups_averaged = []
        for group in self.data:
            coefficients = [c for (c, atom) in group[1:]]
            groups_averaged.append((group[0], np.mean(coefficients), np.std(coefficients)))

        # make `rho_error`, which is a 4x4 array representing the error on each element of the final density matrix,
        rho_error = np.zeros((4, 4), dtype=float)  # changed this from 'object' to 'float' so this might cause some
        # bugs later
        rho_vals = np.zeros((4, 4), dtype=np.complex128)

        for i, j in np.ndindex(rho_error.shape):  # looping over the 4x4
            # create a list of (value, error) tuples
            vals = []
            errs = []
            for (operator, avg, std) in groups_averaged:
                # avg and std are both real numbers!
                op_element = operator[i, j]
                if op_element != 0:
                    vals.append(op_element * avg)
                    errs.append(op_element * std)

            # add the values with proper error propagation
            vals, errs = np.array(vals), np.array(errs)
            val_final = np.sum(vals)  # not mean!
            err_final = np.sqrt(np.sum(np.abs(errs) ** 2))
            rho_error[i, j] = err_final
            rho_vals[i, j] = val_final

        rho_reconst = self.reconstruct_rho()
        rho_vals, rho_error = qt.Qobj(rho_vals, dims=rho_reconst.dims), qt.Qobj(rho_error, dims=rho_reconst.dims)
        rho_cleaned, error_cleaned = clean_dm(rho_vals, abs_trace=abs_trace, errors=rho_error,
                                              positive_diag=positive_diag)
        assert np.array_equal(rho_cleaned, rho_reconst)

        return error_cleaned


def clean_dm(
        rho: qt.Qobj,
        abs_trace: bool = False,
        errors: qt.Qobj = None,
        positive_diag: bool = False
) -> qt.Qobj | tuple[qt.Qobj, qt.Qobj]:
    """
    Removes the "tail" of the density matrix in the diagonal elements.
    Note that this function "normalizes" rho
    Default of abs_trace is False for backwards compatibility (in the old notebook files
     where I used to call clean_dm directly in the notebook instead of inside helper functions).
    But all other calls of clean_dm should have abs_trace set to True (since I think
    it's a better normalization method for a density matrix, since it forces the sum of the diagonal to be 1).
    """
    # newly implemented method: make the diagonal non-negative first
    if positive_diag:
        min_diag = np.min(rho.diag())
        if min_diag < 0:  # make diagonal non-negative
            rho -= op.IDENTITY * min_diag
        assert not np.any(rho.diag() < 0)  # double check diagonal is non-negative

        rho_new = rho / np.sum(rho.diag())
        assert np.isclose(1, rho_new.tr())
        if errors is None:
            return rho_new
        return rho_new, errors / np.sum(rho.diag())

    # most results have used this method so far
    elif abs_trace:  # makes the trace = 1
        abs_diag_sum = np.sum(np.abs(rho.diag()))
        rho_reduced = rho / abs_diag_sum
        rho_no_tail = rho_reduced + op.IDENTITY / 6
        final_multiplier = 1 / np.sum(rho_no_tail.diag())  # force the sum of the diagonal to be 1
        if errors is None:
            return rho_no_tail * final_multiplier
        return rho_no_tail * final_multiplier, errors / abs_diag_sum * final_multiplier

    else:  # uses qutip's ".unit()" function (specifics are complicated)
        if errors is None:
            return (rho.unit() + op.IDENTITY / 6).unit()
        return (rho.unit() + op.IDENTITY / 6).unit(), errors / rho.norm() / (rho.unit() + op.IDENTITY / 6).norm()
