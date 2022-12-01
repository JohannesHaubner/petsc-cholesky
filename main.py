from petsc4py import PETSc
from mpi4py import MPI

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
opts["pc_factor_mat_solver_type"] = "petsc"  # does not run in parallel - needs to be mumps or superlu_dist according to https://gitlab.com/petsc/petsc/-/issues/1182
opts["pc_factor_mat_ordering_type"] = "natural"
PC = solver.getPC()
PC.setFromOptions()
PC.setUp()
chol = PC.getFactorMatrix()

chol.solveBackward(b_petsc, x_petsc)

