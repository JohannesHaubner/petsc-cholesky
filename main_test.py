from petsc4py import PETSc

import scipy.sparse.linalg
import scipy.sparse
from scikits.umfpack import spsolve
import numpy as np

viewer_A = PETSc.Viewer().createBinary('matrix-A.dat', 'r')
viewer_b = PETSc.Viewer().createBinary('matrix-B.dat', 'r')

A_petsc = PETSc.Mat().load(viewer_A)
b_petsc = PETSc.Vec().load(viewer_b)

x_petsc = b_petsc.copy()

solver = PETSc.KSP().create(A_petsc.comm)

solver.setType(PETSc.KSP.Type.PREONLY)
solver.setOperators(A_petsc)

opts = PETSc.Options()
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "petsc"
opts["pc_factor_mat_ordering_type"] = "natural"

PC = solver.getPC()
PC.setFromOptions()
PC.setUp()
chol = PC.getFactorMatrix()

chol.solveForward(b_petsc, x_petsc)
chol.solveBackward(b_petsc, x_petsc)



if A_petsc.comm.size == 1:
    ai, aj, av = A_petsc.getValuesCSR()
    A_org = scipy.sparse.csr_matrix((av, aj, ai)).tocsc()

    LU = scipy.sparse.linalg.splu(
        A_org, permc_spec='NATURAL', diag_pivot_thresh=0)
    chol_scipy = LU.L.dot(scipy.sparse.diags(
        LU.U.diagonal() ** 0.5)).transpose()
    sol = spsolve(chol_scipy, b_petsc.array)

    assert np.allclose(sol, x_petsc.array)
else:
    print(x_petsc.array)

