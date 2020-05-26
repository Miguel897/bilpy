############################ IMPORT ############################

import pdb
import time
import numpy as np
import random
import pandas as pd
import pyomo.environ as pe

##################### LINEAR BILEVEL PROGRAM  ####################

class blp:
  """
  A class to represent a linear bilevel programming problem
  """

  def __init__(self,nvar=10,ncon=5):
    """
    Constructs all the necessary attributes for the following linear bilevel programming problem:
    Min_x  a*x + b*y
    s.t.   C*x <= d 
           x >= 0
           min_y e*y
           s.t.  F*y <= g  
                 H*x + I*y <= j
                 y >= 0
    Parameters
    ----------
    nvar (int, default=10): number of variables
    ncon (int, default=5): number of constraints
    """
    self.nvar = nvar
    self.ncon = ncon
    self.a = [round(abs(random.gauss(0,1)),2) for i in range(nvar)]
    self.b = [round(abs(random.gauss(0,1)),2) for i in range(nvar)]
    self.C = [[round(random.gauss(0,1),2) for i in range(nvar)] for j in range(ncon)]
    self.d = [round(random.gauss(0,1),2) for j in range(ncon)]
    self.e = [round(abs(random.gauss(0,1)),2) for i in range(nvar)]
    self.F = [[round(random.gauss(0,1),2) for i in range(nvar)] for j in range(ncon)]
    self.g = [round(random.gauss(0,1),2) for j in range(ncon)]
    self.H = [[round(random.gauss(0,1),2) for i in range(nvar)] for j in range(ncon)]
    self.I = [[round(random.gauss(0,1),2) for i in range(nvar)] for j in range(ncon)]
    self.j = [round(random.gauss(0,1),2) for j in range(ncon)]

  def gen_model_reg(self):
    """
    Generates the single-level equivalent using KKT with regularization
    Returns
    -------
    m (model): optimization model Pyomo
    """
    # Model
    m = pe.ConcreteModel()
    # Sets
    m.i = pe.Set(initialize=range(self.nvar),ordered=True)
    m.j = pe.Set(initialize=range(self.ncon),ordered=True)
    # Parameters
    m.ep = pe.Param(initialize=10**6,mutable=True)
    # Variables
    m.x = pe.Var(m.i,within=pe.NonNegativeReals)
    m.y = pe.Var(m.i,within=pe.NonNegativeReals)
    m.al = pe.Var(m.j,within=pe.NonNegativeReals)
    m.be = pe.Var(m.j,within=pe.NonNegativeReals)
    m.ga = pe.Var(m.i,within=pe.NonNegativeReals)
    # Objective function
    def obj_rule(m):
      return sum(self.a[i]*m.x[i] for i in m.i) + sum(self.b[i]*m.y[i] for i in m.i)
    m.obj = pe.Objective(rule=obj_rule)
    # Constraints
    def con1_rule(m,j):
      return sum(self.C[j][i]*m.x[i] for i in m.i) <= self.d[j]
    m.con1 = pe.Constraint(m.j,rule=con1_rule)
    def con2_rule(m,j):
      return sum(self.F[j][i]*m.y[i] for i in m.i) <= self.g[j]
    m.con2 = pe.Constraint(m.j,rule=con2_rule)
    def con3_rule(m,j):
      return sum(self.H[j][i]*m.x[i] for i in m.i) + sum(self.I[j][i]*m.y[i] for i in m.i) <= self.j[j]
    m.con3 = pe.Constraint(m.j,rule=con3_rule)
    def con4_rule(m,i):
      return self.e[i] + sum(self.F[j][i]*m.al[j] for j in m.j) + sum(self.I[j][i]*m.be[j] for j in m.j) - m.ga[i] == 0
    m.con4 = pe.Constraint(m.i,rule=con4_rule)
    def con5_rule(m):
      return sum((self.g[j] - sum(self.F[j][i]*m.y[i] for i in m.i))*m.al[j] for j in m.j) + sum((self.j[j] - sum(self.H[j][i]*m.x[i] for i in m.i) - sum(self.I[j][i]*m.y[i] for i in m.i))*m.be[j] for j in m.j) + sum(m.y[i]*m.ga[i] for i in m.i) <= m.ep
    m.con5 = pe.Constraint(rule=con5_rule)
    self.m = m
  
  def gen_model_mip(self):
    """
    Generates the single-level equivalent using KKT with Fortuny-Amat
    Returns
    -------
    m (model): optimization model Pyomo
    """
    # Model
    m = pe.ConcreteModel()
    # Sets
    m.i = pe.Set(initialize=range(self.nvar),ordered=True)
    m.j = pe.Set(initialize=range(self.ncon),ordered=True)
    # Parameters
    m.M = pe.Param(initialize=10**6,mutable=True)
    # Variables
    m.x = pe.Var(m.i,within=pe.NonNegativeReals)
    m.y = pe.Var(m.i,within=pe.NonNegativeReals)
    m.al = pe.Var(m.j,within=pe.NonNegativeReals)
    m.be = pe.Var(m.j,within=pe.NonNegativeReals)
    m.ga = pe.Var(m.i,within=pe.NonNegativeReals)
    m.u1 = pe.Var(m.j,within=pe.Binary)
    m.u2 = pe.Var(m.j,within=pe.Binary)
    m.u3 = pe.Var(m.i,within=pe.Binary)
    # Objective function
    def obj_rule(m):
      return sum(self.a[i]*m.x[i] for i in m.i) + sum(self.b[i]*m.y[i] for i in m.i)
    m.obj = pe.Objective(rule=obj_rule)
    # Constraints
    def con1_rule(m,j):
      return sum(self.C[j][i]*m.x[i] for i in m.i) <= self.d[j]
    m.con1 = pe.Constraint(m.j,rule=con1_rule)
    def con2_rule(m,j):
      return sum(self.F[j][i]*m.y[i] for i in m.i) <= self.g[j]
    m.con2 = pe.Constraint(m.j,rule=con2_rule)
    def con3_rule(m,j):
      return sum(self.H[j][i]*m.x[i] for i in m.i) + sum(self.I[j][i]*m.y[i] for i in m.i) <= self.j[j]
    m.con3 = pe.Constraint(m.j,rule=con3_rule)
    def con4_rule(m,i):
      return self.e[i] + sum(self.F[j][i]*m.al[j] for j in m.j) + sum(self.I[j][i]*m.be[j] for j in m.j) - m.ga[i] == 0
    m.con4 = pe.Constraint(m.i,rule=con4_rule)
    def con5_rule(m,j):
      return self.g[j] - sum(self.F[j][i]*m.y[i] for i in m.i) <= m.u1[j]*m.M
    m.con5 = pe.Constraint(m.j,rule=con5_rule)
    def con6_rule(m,j):
      return m.al[j] <= (1-m.u1[j])*m.M
    m.con6 = pe.Constraint(m.j,rule=con6_rule)
    def con7_rule(m,j):
      return self.j[j] - sum(self.H[j][i]*m.x[i] for i in m.i) - sum(self.I[j][i]*m.y[i] for i in m.i) <= m.u2[j]*m.M
    m.con7 = pe.Constraint(m.j,rule=con7_rule)
    def con8_rule(m,j):
      return m.be[j] <= (1-m.u2[j])*m.M
    m.con8 = pe.Constraint(m.j,rule=con8_rule)
    def con9_rule(m,i):
      return m.y[i] <= m.u3[i]*m.M
    m.con9 = pe.Constraint(m.i,rule=con9_rule)
    def con10_rule(m,i):
      return m.ga[i] <= (1-m.u3[i])*m.M
    m.con10 = pe.Constraint(m.i,rule=con10_rule)
    self.m = m
  
  def solve_ll(self,vector_x,lpsolver='cplex'):
    m = pe.ConcreteModel()
    # Sets
    m.i = pe.Set(initialize=range(self.nvar),ordered=True)
    m.j = pe.Set(initialize=range(self.ncon),ordered=True)
    # Variables
    m.y = pe.Var(m.i,within=pe.NonNegativeReals)
    # Objective function
    def obj_rule(m):
      return sum(self.e[i]*m.y[i] for i in m.i)
    m.obj = pe.Objective(rule=obj_rule)
    # Constraints
    def con1_rule(m,j):
      return sum(self.F[j][i]*m.y[i] for i in m.i) <= self.g[j]
    m.con1 = pe.Constraint(m.j,rule=con1_rule)
    def con2_rule(m,j):
      return sum(self.H[j][i]*vector_x[i] for i in m.i) + sum(self.I[j][i]*m.y[i] for i in m.i) <= self.j[j]
    m.con2 = pe.Constraint(m.j,rule=con2_rule)
    if lpsolver[0:4]=='neos':
      res = pe.SolverManagerFactory('neos').solve(m,opt=pe.SolverFactory(lpsolver[5:]),symbolic_solver_labels=True,tee=True)
    else:
      res = pe.SolverFactory(lpsolver).solve(m,symbolic_solver_labels=True,tee=True)
    return sum(self.a[i]*vector_x[i] + self.b[i]*m.y[i].value for i in m.i)   

  def solve_reg(self,vector_ep = [10**6,10**4,10**2,1,0.1,0.01,0], lpsolver = 'cplex', nlpsolver='ipopt'):
    start = time.time()
    self.gen_model_reg()
    for ep in vector_ep:
      self.m.ep = ep
      if nlpsolver[0:4]=='neos':
        res = pe.SolverManagerFactory('neos').solve(self.m,opt=pe.SolverFactory(nlpsolver[5:]),symbolic_solver_labels=True,tee=True)
      else:
        res = pe.SolverFactory(nlpsolver).solve(self.m,symbolic_solver_labels=True,tee=True)
    of = self.solve_ll(vector_x = [self.m.x[i].value for i in self.m.i],lpsolver=lpsolver)
    return of,time.time()-start

  def solve_mip(self,M = 10**6, lpsolver = 'cplex', mipsolver='ipopt'):
    start = time.time()
    self.gen_model_mip()
    self.m.M = M
    if mipsolver[0:4]=='neos':
      opt = pe.SolverFactory(mipsolver[5:])
      opt.options['mipgap'] = 1e-8
      res = pe.SolverManagerFactory('neos').solve(self.m,opt=opt,symbolic_solver_labels=True,tee=True)
    else:
      opt = pe.SolverFactory(mipsolver)
      opt.options['mipgap'] = 1e-8
      res = opt.solve(self.m,symbolic_solver_labels=True,tee=True)
    of = self.solve_ll(vector_x = [self.m.x[i].value for i in self.m.i],lpsolver=lpsolver)
    return of, time.time()-start


