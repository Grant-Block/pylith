# Step 3: Interseismic Deformation

This example involves a quasi-static simulation for interseismic deformation.
We prescribe aseismic slip (creep) on the bottom of the slab and the deeper portion of the top of the slab; the shallow portion of the top of the slab remains locked.
We use linear Maxwell viscoelastic bulk rheologies in the mantle and deeper part of the slab.
{numref}`fig:example:subduction:3d:step03:diagram` shows the boundary conditions on the domain.

:::{figure-md} fig:example:subduction:3d:step03:diagram
<img src="figs/step03-diagram.*" alt="" width="100%">

Boundary conditions for quasi-static interseismic deformation.
We prescribe aseismic slip (creep) on the bottom of the slab and the deeper portion of the top of the slab; the shallow portion of the top of the slab remains locked.
:::

% Features extracted from simulation parameter files.
```{include} step03_interseismic-synopsis.md
```

## Simulation parameters

The parameters specific to this example are in `step03_interseismic.cfg` and include:

* `pylithapp.metadata` Metadata for this simulation. Even when the author and version are the same for all simulations in a directory, we prefer to keep that metadata in each simulation file as a reminder to keep it up-to-date for each simulation.
* `pylithapp` Parameters defining where to write the output.
* `pylithapp.problem` Parameters for the solution field with displacement and Lagrange multiplier subfields.
* `pylithapp.interfaces` Parameters for the aseismic slip (creep) on the top and bottom of the slab.

For aseismic slip we use the `KinSrcConstRate` kinematic source to prescribe a constant slip rate.
We also adjust the nodesets used for the boundary conditions to remove overlap with the slab to allow the slab to move independently.

```{code-block} console
---
caption: Run Step 3 simulation
---
$ pylith step03_interseismic.cfg mat_viscoelastic.cfg

# The output should look something like the following.
 >> /software/py38-venv/pylith-opt/lib/python3.8/site-packages/pylith/meshio/MeshIOObj.py:44:read
 -- meshiocubit(info)
 -- Reading finite-element mesh
 >> /pylith/libsrc/pylith/meshio/MeshIOCubit.cc:157:void pylith::meshio::MeshIOCubit::_readVertices(pylith::meshio::ExodusII&,
pylith::scalar_array*, int*, int*) const
 -- meshiocubit(info)
 -- Component 'reader': Reading 24824 vertices.
 >> /pylith/libsrc/pylith/meshio/MeshIOCubit.cc:217:void pylith::meshio::MeshIOCubit::_readCells(pylith::meshio::ExodusII&, pyl
ith::int_array*, pylith::int_array*, int*, int*) const
 -- meshiocubit(info)
 -- Component 'reader': Reading 134381 cells in 4 blocks.

# -- many lines omitted --

 >> /pylith/libsrc/pylith/utils/PetscOptions.cc:235:static void pylith::utils::_PetscOptions::write(pythia::journal::info_t&, const char*, const pylith::utils::PetscOptions&)
 -- petscoptions(info)
 -- Setting PETSc options:
fieldsplit_displacement_ksp_type = preonly
fieldsplit_displacement_mg_levels_ksp_type = richardson
fieldsplit_displacement_mg_levels_pc_type = sor
fieldsplit_displacement_pc_type = gamg
fieldsplit_lagrange_multiplier_fault_ksp_type = preonly
fieldsplit_lagrange_multiplier_fault_mg_levels_ksp_type = richardson
fieldsplit_lagrange_multiplier_fault_mg_levels_pc_type = sor
fieldsplit_lagrange_multiplier_fault_pc_type = gamg
ksp_atol = 1.0e-12
ksp_converged_reason = true
ksp_error_if_not_converged = true
ksp_rtol = 1.0e-12
pc_fieldsplit_schur_factorization_type = lower
pc_fieldsplit_schur_precondition = selfp
pc_fieldsplit_schur_scale = 1.0
pc_fieldsplit_type = schur
pc_type = fieldsplit
pc_use_amat = true
snes_atol = 1.0e-9
snes_converged_reason = true
snes_error_if_not_converged = true
snes_monitor = true
snes_rtol = 1.0e-12
ts_error_if_step_fails = true
ts_monitor = true
ts_type = beuler

# -- many lines omitted --

20 TS dt 0.1 time 1.9
    0 SNES Function norm 8.195918512189e+00
    Linear solve converged due to CONVERGED_ATOL iterations 385
    1 SNES Function norm 3.709628561436e-10
  Nonlinear solve converged due to CONVERGED_FNORM_ABS iterations 1
21 TS dt 0.1 time 2.
 >> /software/py38-venv/pylith-opt/lib/python3.8/site-packages/pylith/problems/Problem.py:201:finalize
 -- timedependent(info)
 -- Finalizing problem.
```

The beginning of the output is near the same as in Step 2.
The simulation advances 21 time steps.
The linear solve converged after 385 iterations and the norm of the residual met the absolute convergence tolerance (`ksp_atol`) .
In this simulation the fault interfaces on the top and bottom of the slab occupy a significant fraction of the domain.
As a result, the linear solver requires many more iterations to converge compared to the limited fault interface in Step 2.

## Visualizing the results

The `output` directory contains the simulation output.
Each "observer" writes its own set of files, so the solution over the domain is in one set of files, the boundary condition information is in another set of files, and the material information is in yet another set of files.
The HDF5 (`.h5`) files contain the mesh geometry and topology information along with the solution fields.
The Xdmf (`.xmf`) files contain metadata that allow visualization tools like ParaView to know where to find the information in the HDF5 files.
To visualize the data using ParaView or Visit, load the Xdmf files.

In {numref}`fig:example:subduction:3d:step03:solution` we use ParaView to visualize the x displacement field using the `viz/plot_dispwarp.py` Python script.
We start ParaView from the `examples/subduction-3d` directory and then run the `viz/plot_dispwarp.py` Python script as described in {ref}`sec-paraview-python-scripts`.

:::{figure-md} fig:example:subduction:3d:step03:solution
<img src="figs/step03-solution.*" alt="Solution for Step 3. The colors indicate the magnitude of the x displacement, and the deformation is exaggerated by a factor of 1000." width="100%"/>

Solution for Step 3.
The colors of the shaded surface indicate the magnitude of the x displacement, and the deformation is exaggerated by a factor of 1000.
:::
