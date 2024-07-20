from dolfin import *
import numpy as np



# Define domain
L = 5
H = 20

xden = 60
yden = 200

mesh = RectangleMesh(Point(-1, 0), Point(L, H), xden, yden ) 
x = SpatialCoordinate(mesh)
V = VectorFunctionSpace(mesh,"CG",1)
u = interpolate(Expression(("0", "((x[1]*(exp(x[1]/20) -1 ))/(exp(1)-1)) - x[1]"), degree=1),V)
w = interpolate(Expression(("x[0]< 1 ? -(0.5)*(x[0]-1) :0 ","0"),  degree=1),V)

ALE.move(mesh,u)
ALE.move(mesh,w)


# # Define Taylor--Hood function space W
V = VectorElement("Lagrange", triangle, 2)
Q = FiniteElement("Lagrange", triangle, 1)


W = FunctionSpace(mesh, MixedElement([V, Q]))

# Define Function and TestFunction(s)
w = Function(W)

(v, q) = split(TestFunction(W))

x = SpatialCoordinate(mesh)


sigma = '(sqrt(1 + exp(-2*x[0])))'
Uw = '(x[0] + std::log((1 + ' + sigma + ' )/(1 + sqrt(2)) ) + sqrt(2) -' + sigma  + ')'
Uwp = '(1 + exp(-2*x[0])/'+ sigma + '*(1 - 1/( ' + sigma + ' + 1)) ) '
y = '(x[1]/' + sigma + ')'
fp = 'exp( - ' + y + ')'
f = '(1 - ' + fp + ')'

u_bl = Uw + '/' + sigma + '*' + fp
v_bl = '-' + Uw + '*exp(-2*x[0])/(pow(' + sigma + ', 2))*' +  y + '*' + fp +  \
       '-' +  Uwp + '*' + f


u_ex = Expression((u_bl, v_bl), degree=2)

out = 'near(x[0],' + str(L) +  ')'
FS = 'near(x[1],' + str(H) + ')'
sheet = 'near(x[1], 0.0)'
inl = 'near(x[0], 0.0)'
bcu_inflow = DirichletBC(W.sub(0).sub(0), 0,  inl) # change (0,1) to u_in for die swell
bcu_FS = DirichletBC(W.sub(0), u_ex, FS) 
bcu_outflow = DirichletBC(W.sub(0), u_ex, out)
bcu_sheet = DirichletBC(W.sub(0), u_ex, sheet)
bcp_FS =  DirichletBC(W.sub(1), 0,FS)

bcs = [bcu_inflow,  bcu_outflow, bcu_sheet, bcu_FS, bcp_FS]





w = interpolate(Expression((u_bl,v_bl,'0*x[0]'), degree=2) ,W)
(u, p) = split(w)


ep = .2


# y = y.reshape(yden+1,xden+1)
# y = y[:,0]


# u_0 = np.exp(-y)     # at eta = 1xden+
# v_0  = np.exp(-y) -1
# p_0 = -np.exp(-2*y)/2
# define our stress terms
s = Expression('exp(-x[0])', degree = 2)

s_p = -s
s_pp = s

# define our stress terms
sig_x  = - ep**2*p + ep**2*( 2*u[0].dx(0) - 2/ep*s_p*u[0].dx(1) )


sig_xy = ep*u[0].dx(1) + ep**3*u[1].dx(0) + ep**2*(s_pp*u[0] + s_p*u[0].dx(0))

sig_yy = -ep**2*p + ep**2*(2*u[1].dx(1) + 2/ep*s_p*u[0].dx(1))

# # "Convective" Terms
x_convective = u[0]*u[0].dx(0) + u[1]*u[0].dx(1)
y_convective = ep*(u[0]*u[1].dx(0) + u[1]*u[1].dx(1) ) + s_p*x_convective + s_pp*u[0]**2 

ce = (q*(u[0].dx(0) + u[1].dx(1)))*dx

xm = ( v[0]*x_convective + (v[0].dx(0)-  1/ep*s_p*v[0].dx(1)  )*sig_x +  1/ep*v[0].dx(1)*sig_xy ) *dx 

ym = (v[1]*y_convective + ( v[1].dx(0) - 1/ep*v[1].dx(1)*s_p  )*sig_xy + 1/ep*v[1].dx(1)*sig_yy )*dx  #\


F = ce + xm + ym



solve(F == 0, w, bcs)

# Plot solutions
(u, p) = w.split()



xy = np.array(mesh.coordinates())


x = xy[:,0]
y = xy[:,1]

uv = np.array([u(pt) for pt in xy])
P_ = np.array([p(pt) for pt in xy])




X = x.reshape(yden+1,xden+1)
Y = y.reshape(yden+1,xden+1)
u_ = uv[:,0]
v_ = uv[:,1]

U = u_.reshape(yden+1,xden+1)
V = v_.reshape(yden+1,xden+1)
P = P_.reshape(yden+1,xden+1)


tableur = X
np.savetxt('x.dat',tableur)

tableur = Y
np.savetxt('y.dat',tableur)

tableur = U
np.savetxt('u.dat',tableur)

tableur = V
np.savetxt('v.dat',tableur)

tableur = P
np.savetxt('p.dat',tableur)




