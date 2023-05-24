#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 23:35:43 2023

@author: julio
"""

from fenics import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create mesh and define function space
nx, ny = 300, 300
mesh = RectangleMesh(Point(0, 0), Point(2, 1), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
def boundary_x0(x, on_boundary):
    return on_boundary and near(x[1], 0)

bc_x0 = DirichletBC(V, Expression("x[0]", degree=2), boundary_x0)

# Define boundary condition at x = 1
def boundary_x1(x, on_boundary):
    return on_boundary and near(x[1], 1)

bc_x1 = DirichletBC(V, Expression("exp(1)*x[0]", degree=2), boundary_x1)

# Define source term
f = Expression("x[0]*exp(x[1])",
               degree=1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, [bc_x0,bc_x1])

# Get the coordinates and values of the solution for plotting
coords = V.tabulate_dof_coordinates()
values = u.compute_vertex_values(mesh)

# Create a 3D plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

# The x, y, and z coordinates are reshaped to form a grid for plotting
X = coords[:,0].reshape((nx+1, ny+1))
Y = coords[:,1].reshape((nx+1, ny+1))
Z = values.reshape((nx+1, ny+1))

# Plot the surface
#ax.plot_surface(X, Y, Z)

c=plot(u, title='Solución EDP Elíptica')
plt.colorbar(c)

# Set labels and title
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('u')
#ax.set_title('Solution to the Poisson equation')

plt.show()
