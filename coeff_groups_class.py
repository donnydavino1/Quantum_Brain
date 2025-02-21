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
            if operator == group[0]:
                group.append((c, atom))
                return
            elif -operator == group[0]:
                group.append((-c, atom))
                return
        self.data.append([operator, (c, atom)])

    def reconstruct_rho(self, abs_trace=True, positive_diag=False, add_identity=True, exclude_i=None) -> qt.Qobj:
        """
        Given a 'coefficient groups', construct a density matrix.
        The parameters `abs_trace`, `positive_diag`, and `add_identity` are passed on to `clean_dm()` and these
        parameters are explained in more detail in the docstring of `clean_dm()`.


        """
        # Make sure the groups cover all 15 coefficients required to reconstruct a density matrix
        assert len(self.data) == 15, "number of groups should be 15!"

        rho = qt.Qobj(np.zeros(self.data[0][0].shape), dims=self.data[0][0].dims)
        for group in self.data:
            coefficients = [c for (c, atom) in group[1:]]
            rho += group[0] * np.mean(coefficients)
        rho = clean_dm(rho, abs_trace=abs_trace, positive_diag=positive_diag, add_identity=add_identity)
        return rho

    def get_error(self, abs_trace=True, positive_diag=False, add_identity=True) -> qt.Qobj:
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
                                              positive_diag=positive_diag, add_identity=add_identity)
        assert np.array_equal(rho_cleaned, rho_reconst)

        return error_cleaned


def clean_dm(
        rho: qt.Qobj,
        abs_trace: bool = False,
        errors: qt.Qobj = None,
        positive_diag: bool = False,
        add_identity: bool = True,
) -> qt.Qobj | tuple[qt.Qobj, qt.Qobj]:
    """
    The final processing step to 'clean up' the density matrix (rho) from tomography.
    The behavior of this function varies widely depending on the given parameters,
    so pay careful attention to their values.

    The cleanup process is tricky since the resulting density matrix from tomography contains negative values
    in the diagonal and the trace is always 0, which obviously should not be true for a real density matrix.
    (This is because the tomography's basis product operators all have trace 0)

    2024 December: Added the option "add_identity" (which was always True previously). If this is set to false,
    it does not add any identity to the density matrix (and thus doesn't try to get rid of the "tail" of the diagonal).
    This is useful when I'm trying to compare the raw measured density matrix from the tomography to the theoretical
    density matrix with a tail.

    :param rho:
        The density matrix to clean.
    :param abs_trace:
        Divides the diagonal by the "absolute trace".
        I've defined the "absolute trace" as the sum of the absolute of the diagonal elements.
        Allows the negative values in the diagonal.
        Additionally, adds identity to the diagonal depending on the `add_identity` parameter
        (further details in the `add_identity` docstring).

        Usually set to True, and this is the 'default' behavior of this function. However, the default value of the
        parameter is set to False for backwards compatibility (in the old notebook files where I used to call
        clean_dm directly in the notebook instead of inside helper functions).
    :param errors:
        The error matrix, where each element represents the error on the corresponding element of rho.
        If `errors` is provided, does proper error propagation then return a tuple (rho, errors).
    :param positive_diag:
        If True, forces all the diagonal elements to be non-negative by subtracting the most negative diagonal
        element to all the diagonal elements.
        Since all diagonal elements are now non-negative, we no longer have to worry about trace vs absolute trace,
        and can reliably use regular trace for normalization.
    :param add_identity:
        Only relevant if `abs_trace` is True. This value was always True by default in the past.
        If `add_identity` is True the value (identity / 6) (*this value should be re-examined?*) then
        re-normalizes by the trace (NOT the absolute trace!).
        Perhaps should experiment with normalizing with absolute trace here instead?
    :return:
        the processed rho density matrix Qobj.
        If the `errors` matrix is provided, returns a tuple of Qobj's instead: (rho, errors).
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
        error_multiplier = 1 / abs_diag_sum

        # In the past, this was always True. Now I'm adding a feature where I can turn this off so
        # I can make measured & theoretical matrices without identities added to them.
        if add_identity:
            rho_reduced = rho_reduced + op.IDENTITY / 6
            # force the sum of the diagonal to be 1 again
            rho_reduced = rho_reduced / np.sum(rho_reduced.diag())
            error_multiplier = error_multiplier / np.sum(rho_reduced.diag())

        if errors is None:
            return rho_reduced
        return rho_reduced, errors * error_multiplier

    else:  # uses qutip's ".unit()" function (specifics are complicated)
        if errors is None:
            return (rho.unit() + op.IDENTITY / 6).unit()
        return (rho.unit() + op.IDENTITY / 6).unit(), errors / rho.norm() / (rho.unit() + op.IDENTITY / 6).norm()
