from dolfin import UnitSquareMesh,  assemble, FunctionSpace, inner, dx, TestFunction, TrialFunction, as_backend_type, Function, Expression, solve
from petsc4py import PETSc
import scipy.sparse.linalg
import scipy.sparse
from scikits.umfpack import spsolve
import numpy as np

mesh = UnitSquareMesh(5, 5)
V = FunctionSpace(mesh, "Lagrange", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(u, v) * dx
A = assemble(a)
A_petsc = as_backend_type(A).mat()

b = Function(V)
b.interpolate(Expression("x[0]", degree=1))
b_petsc = as_backend_type(b.vector()).vec()

viewer_A = PETSc.Viewer().createBinary('matrix-A.dat', 'w')
viewer_b = PETSc.Viewer().createBinary('matrix-B.dat', 'w')
A_petsc.view(viewer_A)
b_petsc.view(viewer_b)
