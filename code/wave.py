#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 23:02:25 2023

@author: julio
"""

from fenics import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create mesh and define function space
nx, nt = 30, 30
#mesh = UnitSquareMesh(nx, nt)
#mesh=IntervalMesh(100, 0, 1)
#nx, ny = 300, 300
mesh = RectangleMesh(Point(0, 0), Point(1, 3), nx, nt)
V = FunctionSpace(mesh, 'Lagrange', 2)

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary# and (x[0]==0 or x[0]==1)

# Define boundary condition
def boundary_x0(x, on_boundary):
    return on_boundary and near(x[0], 0)

bc_x0 = DirichletBC(V, Expression("0", degree=2), boundary_x0)

def boundary_x1(x, on_boundary):
    return on_boundary and near(x[0], 1)

bc_x1 = DirichletBC(V, Expression("0", degree=2), boundary_x1)

bc = DirichletBC(V, Constant(0), boundary)

# Define initial condition
u_1 = interpolate(Expression('sin(pi*x[1])', degree=2), V)
#u_1 = interpolate(Expression('0', degree=2), V)
u_0 = interpolate(Constant(0.0),V) # initial condition 

# Time-stepping parameters
T = 3.0            # total simulation time
dt = 0.001          # time step size

c=2
num_steps = int(T/dt)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

#a = u*v*dx + dt*dt*4*inner(grad(u), grad(v))*dx
a = inner(u,v)*dx + dt*dt*c*c*inner(grad(u), grad(v))*dx
#L = 2*u_1*v*dx - u_0*v*dx
L = 2*inner(u_1,v)*dx- inner(u_0, v)*dx
u = Function(V)


solutions = []
# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    
    #A, b = assemble_system(a, L, bc)
    # Compute solution
    solve(a == L, u,bc)#[bc_x0,bc_x1])
    
    #solve(A,u.vector(),b)

    # Update previous solution
    u_0.assign(u_1)
    u_1.assign(u)
    
    solutions.append(u.copy())
    
    t += dt

# Save solution to file
vtkfile = File("wave.pvd")
vtkfile << u
c=plot(u, title='Solución numérica Ecuación de Onda')

