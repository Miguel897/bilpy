import pdb
from bilpy import blp

blp1 = blp(nvar=50,ncon=25)

#blp1.gen_model_mip()

of1,time1 = blp1.solve_reg(vector_ep = [10**6,10**4,10**2,1,0.1,0.01,0], lpsolver='cplex', nlpsolver='ipopt')
of2,time2 = blp1.solve_mip(M=1000, lpsolver = 'cplex', mipsolver='cplex')

pdb.set_trace()
