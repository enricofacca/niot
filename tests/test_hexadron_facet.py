from firedrake import *

mesh = UnitCubeMesh(2,2,2,hexahedral=True)

W = FunctionSpace(mesh, "DQ", 0)
u = Function(W)
v = TestFunction(W)


h = CellSize(W.mesh())
h_avg = (h('+') + h('-'))/2

a_dg = - 4.0/h_avg * jump(u)* jump(v) * dS
A = assemble(a_dg)
