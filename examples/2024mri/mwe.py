import firedrake
from firedrake import FunctionSpace, Function, TrialFunction, TestFunction, Constant, dot, grad, dx, conditional, And, Or, solve, VTKFile, UnitSquareMesh
from firedrake.mesh import MeshGeometry, MeshTopology, make_mesh_from_mesh_topology, ExtrudedMeshTopology, VertexOnlyMeshTopology
from firedrake.cython import dmcommon
from firedrake.cython.dmcommon import DistributedMeshOverlapType
import numbers
from collections.abc import Sequence

from firedrake import SpatialCoordinate
from firedrake.utils import as_cstr, IntType, RealType

from firedrake import DirichletBC

DEFAULT_MESH_NAME = "RelabeledMesh"

def _generate_default_mesh_topology_name(mesh_name):
    return f"{mesh_name}_topology"

def MyRelabeledMesh(mesh, indicator_functions, subdomain_ids, boundary_only=False, **kwargs):
    """Construct a new mesh that has new subdomain ids.

    :arg mesh: base :class:`~.MeshGeometry` object using which the
        new one is constructed.
    :arg indicator_functions: list of indicator functions that mark
        selected entities (cells or facets) as 1; must use
        "DP"/"DQ" (degree 0) functions to mark cell entities and
        "P" (degree 1) functions in 1D or "HDiv Trace" (degree 0) functions
        in 2D or 3D to mark facet entities.
        Can use "Q" (degree 2) functions for 3D hex meshes until
        we support "HDiv Trace" elements on hex.
    :arg subdomain_ids: list of subdomain ids associated with
        the indicator functions in indicator_functions; thus,
        must have the same length as indicator_functions.
    :kwarg name: optional name of the output mesh object.
    """
    import firedrake.function as function

    if not isinstance(mesh, MeshGeometry):
        raise TypeError(f"mesh must be a MeshGeometry, not a {type(mesh)}")
    tmesh = mesh.topology
    if isinstance(tmesh, VertexOnlyMeshTopology):
        raise NotImplementedError("Currently does not work with VertexOnlyMesh")
    elif isinstance(tmesh, ExtrudedMeshTopology):
        raise NotImplementedError("Currently does not work with ExtrudedMesh; use RelabeledMesh() on the base mesh and then extrude")
    if not isinstance(indicator_functions, Sequence) or \
       not isinstance(subdomain_ids, Sequence):
        raise ValueError("indicator_functions and subdomain_ids must be `list`s or `tuple`s of the same length")
    if len(indicator_functions) != len(subdomain_ids):
        raise ValueError("indicator_functions and subdomain_ids must be `list`s or `tuple`s of the same length")
    if len(indicator_functions) == 0:
        raise RuntimeError("At least one indicator function must be given")
    for f in indicator_functions:
        if not isinstance(f, function.Function):
            raise TypeError(f"indicator functions must be instances of function.Function: got {type(f)}")
        if f.function_space().mesh() is not mesh:
            raise ValueError(f"indicator functions must be defined on {mesh}")
    for subid in subdomain_ids:
        if not isinstance(subid, numbers.Integral):
            raise TypeError(f"subdomain id must be an integer: got {subid}")
    name1 = kwargs.get("name", DEFAULT_MESH_NAME)
    plex = tmesh.topology_dm
    # Clone plex: plex1 will share topology with plex.
    plex1 = plex.clone()
    plex1.setName(_generate_default_mesh_topology_name(name1))
    # Remove pyop2 labels.
    plex1.removeLabel("pyop2_core")
    plex1.removeLabel("pyop2_owned")
    plex1.removeLabel("pyop2_ghost")
    # Do not remove "exterior_facets" and "interior_facets" labels;
    # those should be reused as the mesh has already been distributed (if size > 1).
    for label_name in [dmcommon.CELL_SETS_LABEL, dmcommon.FACE_SETS_LABEL]:
        if not plex1.hasLabel(label_name):
            plex1.createLabel(label_name)
    for f, subid in zip(indicator_functions, subdomain_ids):
        elem = f.topological.function_space().ufl_element()
        if elem.reference_value_shape != ():
            raise RuntimeError(f"indicator functions must be scalar: got {elem.reference_value_shape} != ()")
        if elem.family() in {"Discontinuous Lagrange", "DQ"} and elem.degree() == 0:
            # cells
            height = 0
            dmlabel_name = dmcommon.CELL_SETS_LABEL
        elif (elem.family() == "HDiv Trace" and elem.degree() == 0 and mesh.topological_dimension() > 1) or \
                (elem.family() == "Lagrange" and elem.degree() == 1 and mesh.topological_dimension() == 1) or \
                (elem.family() == "Q" and elem.degree() == 2 and mesh.topology.ufl_cell().cellname() == "hexahedron"):
            # facets
            height = 1
            dmlabel_name = dmcommon.FACE_SETS_LABEL
        else:
            raise ValueError(f"indicator functions must be 'DP' or 'DQ' (degree 0) to mark cells and 'P' (degree 1) in 1D or 'HDiv Trace' (degree 0) in 2D or 3D to mark facets: got (family, degree) = ({elem.family()}, {elem.degree()})")
        # Clear label stratum; this is a copy, so safe to change.
        plex1.clearLabelStratum(dmlabel_name, subid)
        dmlabel = plex1.getLabel(dmlabel_name)

        print(f.dat.data_ro_with_halos.size)

        plex1.markBoundaryFaces("boundary_faces")
        coords = plex1.getCoordinates()
        coord_sec = plex1.getCoordinateSection()
        
        if boundary_only and dmlabel_name == dmcommon.FACE_SETS_LABEL:
            group = "boundary_faces"
            section = f.topological.function_space().dm.getSection()
            if plex1.getStratumSize(group, 1) > 0:
                boundary_faces = plex1.getStratumIS(group, 1).getIndices()
                for facet_point in boundary_faces:
                    offset = section.getOffset(facet_point)   
                    if f.dat.data_ro_with_halos[offset] > 0.5:
                        face_coords = plex1.vecGetClosure(coord_sec, coords, facet_point)
                        plex1.setLabelValue(dmlabel_name, facet_point, subid)
            plex1.removeLabel("boundary_faces")
        elif boundary_only and dmlabel_name == dmcommon.CELL_SETS_LABEL:
            group = "boundary_faces"
            section = f.topological.function_space().dm.getSection()
            if plex1.getStratumSize(group, 1) > 0:
                boundary_faces = plex1.getStratumIS(group, 1).getIndices()
                for facet_point in boundary_faces:
                    cells = plex1.getSupport(facet_point)
                    cell_point = cells[0]
                    offset = section.getOffset(cell_point)
                    if f.dat.data_ro_with_halos[offset] > 0.5:
                        face_coords = plex1.vecGetClosure(coord_sec, coords, facet_point)
                        print(facet_point, face_coords)
                        print(f"Setting label for facet_point {facet_point} to {subid}")
                        plex1.setLabelValue(dmcommon.FACE_SETS_LABEL, facet_point, subid)
            plex1.removeLabel("boundary_faces")

        else:
            section = f.topological.function_space().dm.getSection()
            dmcommon.mark_points_with_function_array(plex, section, height, f.dat.data_ro_with_halos.real.astype(IntType), dmlabel, subid)
    
    distribution_parameters_noop = {"partition": False,
                                    "overlap_type": (DistributedMeshOverlapType.NONE, 0)}
    reorder_noop = None
    tmesh1 = MeshTopology(plex1, name=plex1.getName(), reorder=reorder_noop,
                          distribution_parameters=distribution_parameters_noop,
                          perm_is=tmesh._dm_renumbering,
                          distribution_name=tmesh._distribution_name,
                          permutation_name=tmesh._permutation_name,
                          comm=tmesh.comm)
    return make_mesh_from_mesh_topology(tmesh1, name1)

mesh = UnitSquareMesh(40, 40)#, reorder=False)
x, y = SpatialCoordinate(mesh)

# Define indicator function suitable for RelabeledMesh
W = FunctionSpace(mesh, "HDiv Trace", 0)
W = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
indicator = Function(W, name="boundary")
indicator.interpolate(conditional(And(x > 0.25, y > 0.45), 1, 0))
relabeled_mesh = MyRelabeledMesh(mesh, [indicator], [99], boundary_only=True)

# Verify with a Poisson Solve
V = FunctionSpace(relabeled_mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
DG0 = FunctionSpace(relabeled_mesh, "DG", 0)
#indicator_new = Function(DG0, name="boundary_new")
#indicator_new.interpolate(indicator)

a = dot(grad(u), grad(v)) * dx
L = v * dx
bc = DirichletBC(V, 0.0, 99)

u_sol = Function(V,name="u_sol")

# Apply BC only on the newly marked region     
try:
    solve(a == L, u_sol, bcs=[bc])
    print("Solve successful using new boundary ID.")
except Exception as e:
    print(f"Solve failed: {e}")

outfile = VTKFile("solution.pvd")
outfile.write(u_sol)#, indicator_new)


#indicator.interpolate(conditional(Or(And(x > 0.5,abs(y-1)<1e-4), And(abs(x-1)<1e-4,y > 0.5)), 1, 0))
