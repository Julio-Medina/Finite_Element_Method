#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:36:35 2023

@author: julio
"""

from fenics import *
import matplotlib.pyplot as plt

# Create mesh and define function space
nx, ny = 30, 30
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)


def boundary_x0(x, on_boundary):
    return on_boundary and near(x[0], 0)

bc_x0 = DirichletBC(V, Expression("0", degree=2), boundary_x0)

# Define boundary condition at x = 1
def boundary_x1(x, on_boundary):
    return on_boundary and near(x[0], 1)

bc_x1 = DirichletBC(V, Expression("0", degree=2), boundary_x1)

# Define initial value
u_0 = Expression('sin(pi*x[0])',
                 degree=2, a=5)
u_n = interpolate(u_0, V)

# Define parameters
T = 1.0            # final time
num_steps = 500     # number of time steps
dt = T / num_steps # time step size

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0

for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u,[bc_x0,bc_x1])

    # Plot solution
    

    # Update previous solution
    u_n.assign(u)
c=plot(u, title="Solución Ecuación Parabólica")
#plt.colorbar(c)
plot(mesh)
# Show the plot at the end of the simulation
plt.show()
