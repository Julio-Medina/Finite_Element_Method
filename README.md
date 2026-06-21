# Finite Element Method

This repository presents a theoretical introduction to the **Finite Element Method (FEM)** and compares implementations in **Wolfram Mathematica** and **Python/FEniCS** for three canonical classes of partial differential equations:

- elliptic equations, represented by the Poisson equation;
- parabolic equations, represented by the heat equation;
- hyperbolic equations, represented by the wave equation.

The accompanying report develops the variational and weak-form perspectives, discusses triangular finite elements and mesh generation, and compares the numerical implementations and their memory use.

## Repository structure

```text
.
├── README.md
├── code/
│   ├── elliptic.py
│   ├── parabolic.py
│   ├── wave.py
│   ├── Elliptic_equation.nb
│   ├── Parabolic_equation.nb
│   └── Wave_equation.nb
├── report/
│   ├── Finite_Element_Method.tex
│   └── ... figure files used by the report
└── report_EN/
    └── Finite_Element_Method_EN.tex
```

The English report reuses the figures stored in `report/`, so its `\includegraphics` paths point to `../report/`.

## Problems studied

### Elliptic problem

The report considers the two-dimensional Poisson equation

$$\nabla^2 u(x,y)=x e^y, \qquad 0<x<2, \quad 0<y<1,$$

with Dirichlet boundary data consistent with the analytical function $u(x,y)=x e^y$. Both a rectangular domain and a more complex region with an excluded circular subdomain are explored in Mathematica.

### Parabolic problem

The parabolic example is the one-dimensional heat equation

$$\frac{\partial u}{\partial t}-\frac{\partial^2u}{\partial x^2}=0, \qquad 0<x<1, \quad t\geq 0,$$

with homogeneous Dirichlet conditions and the initial profile

$$u(x,0)=\sin(\pi x).$$

### Hyperbolic problem

The hyperbolic example is the wave equation

$$\frac{\partial^2u}{\partial t^2}-4\frac{\partial^2u}{\partial x^2}=0, \qquad 0<x<1, \quad t>0,$$

with fixed endpoints, initial displacement $u(x,0)=\sin(\pi x)$, and zero initial velocity.

## Numerical implementations

### Python/FEniCS

The Python scripts use the legacy `fenics` interface and define finite-element spaces, boundary conditions, variational forms, and numerical solutions.

Required Python packages include:

```text
fenics
matplotlib
numpy
```

Run a script from the repository root with, for example:

```bash
python code/elliptic.py
python code/parabolic.py
python code/wave.py
```

The wave script also writes a `wave.pvd` file that can be opened with visualization software such as ParaView.

### Wolfram Mathematica

The Mathematica notebooks were created with Mathematica 12 and use the finite-element functionality provided through `NDSolve` and the <code>NDSolve`FEM`</code> context. Open the `.nb` files in Mathematica and evaluate the cells in sequence.

## Important implementation notes

1. **Elliptic sign convention.** The report states $\nabla^2u=x e^y$, while the FEniCS code uses the bilinear and linear forms `a = dot(grad(u), grad(v))*dx` and `L = f*v*dx`. With homogeneous test functions, that variational form corresponds to $-\nabla^2u=f$. To match the PDE and boundary data in the report, use `L = -f*v*dx`; alternatively, redefine the source term with the opposite sign.

2. **Parabolic spatial dimension.** The report describes a one-dimensional heat equation, but `parabolic.py` constructs a `UnitSquareMesh`, introducing a second spatial coordinate. An `IntervalMesh` would represent the reported one-dimensional problem more directly.

3. **Wave formulation.** The `wave.py` script constructs a two-dimensional rectangle in $(x,t)$ and repeatedly solves on that full mesh. This differs from a standard semidiscrete FEM time integrator, where the spatial domain is discretized first and the resulting system is advanced in time.

These notes do not prevent the files from serving as educational examples, but they are relevant when comparing the code directly with the equations stated in the report.

## Compiling the reports

From the Spanish report directory:

```bash
cd report
pdflatex Finite_Element_Method.tex
pdflatex Finite_Element_Method.tex
```

From the English report directory:

```bash
cd report_EN
pdflatex Finite_Element_Method_EN.tex
pdflatex Finite_Element_Method_EN.tex
```

Running LaTeX twice resolves cross-references. The figure files referenced by the reports must be present in `report/`.

## Main topics covered in the report

- functional minimization and the Rayleigh--Ritz viewpoint;
- piecewise-linear and bilinear basis functions;
- triangular elements and nodal basis functions;
- strong and weak formulations of the Poisson equation;
- essential Dirichlet and natural Neumann boundary conditions;
- domain tessellation and mesh-generation strategies;
- Mathematica and FEniCS implementation comparisons;
- basic statistical analysis of numerical approximation results.

## Author

**BSc. Julio A. Medina**  
University of San Carlos of Guatemala  
School of Physical Sciences and Mathematics  
Master's Program in Physics

## References

The full bibliography is included in the LaTeX report. Principal references include Burden and Faires on numerical analysis, the FEniCS project papers and monograph, and Jackson's *Classical Electrodynamics*.
