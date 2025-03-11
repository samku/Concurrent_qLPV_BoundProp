import sys
import os
base_dir = os.path.dirname(os.path.dirname(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from imports import *
from qLPV_model import qLPV_model

#Load model
current_directory = Path(__file__).parent.parent
file_path = current_directory / "identification_results/oscillator_lowNoise/dataset.pkl"
with open(file_path, 'rb') as f:
    dataset = pickle.load(f)

file_path = current_directory / "identification_results/oscillator_lowNoise/concurrent_new_005.pkl"
with open(file_path, 'rb') as f:
    models = pickle.load(f)

#Extract system
system = dataset['system']
model_LPV_concur = models['model_LPV_concur']
RCI_concur = models['RCI_concur']
dt = system.dt if hasattr(system, 'dt') else 1.0

#Extract model parameters
sim_parameters = model_LPV_concur.copy()
sim_parameters['sizes'] = models['sizes']
sim_parameters['W'] = RCI_concur['W']
sim_parameters['u_scaler'] = dataset['u_scaler']
sim_parameters['y_scaler'] = dataset['y_scaler']
sim_parameters['HU'] = dataset['constraints']['HU']
sim_parameters['hU'] = RCI_concur['hU_modified']
sim_parameters['HY'] = dataset['constraints']['HY']
sim_parameters['hY'] = dataset['constraints']['hY']
activation = models['activation']
model = qLPV_model(sim_parameters, activation = activation)

A = model.A
B = model.B
C = model.C
L = model.L
print(L)
HY = model.HY
hY = model.hY
HU = model.HU
hU = model.hU
nx = model.nx
ny = model.ny



nu = model.nu
F = RCI_concur['F']
V = RCI_concur['V']
yRCI = np.array(RCI_concur['yRCI'])
m_bar = len(V)

#Compute tracking bounds
tracking_output = 1
C = C[0] #Hardcoded for single output
xRCI_vert = np.zeros((m_bar, nx))
xRCI_vert_LTI = np.zeros((m_bar, nx))
xRCI_vert_LPV_init = np.zeros((m_bar, nx))
yRCI_vert = np.zeros((m_bar, ny))
yRCI_vert_LTI = np.zeros((m_bar, ny))
yRCI_vert_LPV_init = np.zeros((m_bar, ny))
for k in range(m_bar):
    xRCI_vert[k] = V[k] @ yRCI
    yRCI_vert[k] = C @ V[k] @ yRCI

y_track_max = model.y_scaler[0] + np.max(yRCI_vert)/model.y_scaler[1]
y_track_min = model.y_scaler[0] + np.min(yRCI_vert)/model.y_scaler[1]

Hcon_plant = HY * model.y_scaler[1]
hcon_plant = hY + Hcon_plant @ model.y_scaler[0]
Hcon = Polytope(A=Hcon_plant, b=hcon_plant)
Y_con_vert_plant = Hcon.V
y_max_con = np.max(Y_con_vert_plant, axis = 0)
y_min_con = np.min(Y_con_vert_plant, axis = 0)

#Compute input bounds
u_bounds_plant = model.u_scaler[0] + model.hU[0:model.nu]/model.u_scaler[1]

#Disturbance bounds
hW = np.array([model.W[0]+model.W[1], model.W[0]-model.W[1]]).reshape(1,-1)[0]
w_bounds_ub = model.y_scaler[0] + (model.W[0]+model.W[1])/model.y_scaler[1]
w_bounds_lb = model.y_scaler[0] + (model.W[0]-model.W[1])/model.y_scaler[1]

opts = {"verbose": False,  
        "ipopt.print_level": 0,  
        "print_time": 0 }

#Simulate in closed loop
N_sim = 200
N_MPC = 5

x_plant = np.zeros((N_sim+1,system.nx_plant))
system.state = x_plant[0]
u_plant = np.zeros((N_sim,system.nu_plant))
y_plant = np.zeros((N_sim,model.ny))
parameters_sim = np.zeros((N_sim, model.nq))

x_model = np.zeros((N_sim+1,model.nx))
x_next_bad = np.zeros((N_sim+1,model.nx))
y_model = np.zeros((N_sim,model.ny))
p_model = np.zeros((N_sim,model.nq))
dy_model = np.zeros((N_sim,model.ny))

ref_y = np.zeros((N_sim, 1))

for t in range(N_sim):
    #print(t)
    y_model[t] = model.output(x_model[t])
    y_plant[t] = system.output(x_plant[t], 0.)
    dy_model[t] = y_plant[t] - y_model[t]

    if t%30 == 0 or t == 0:
        #Reference in plant space
        ref_y[t] = 1.2*(y_track_min + (y_track_max-y_track_min)*np.random.rand(1))
    else:
        ref_y[t] = ref_y[t-1]
    if t<=30:
        ref_y[t] = y_track_min*1.1
    elif t>30 and t<=40:
        ref_y[t] = y_track_max*1.1
    

    #MPC
    if N_MPC>1:
        opti = ca.Opti()
        u_MPC = opti.variable(model.nu, N_MPC)
        x_MPC = opti.variable(model.nx, N_MPC+1)
        x_next_1 = model.dynamics(x_model[t], \
                                model.u_scaler[0] + u_MPC[:,0]/model.u_scaler[1], \
                                y_plant[t])
        cost = 0.
        for i in range(N_MPC):
            if i == 0:
                opti.subject_to(x_MPC[:,i] == x_model[t])
            elif i == 1:
                opti.subject_to(x_MPC[:,i] == x_next_1)
            else:
                opti.subject_to(x_MPC[:,i] == model.dynamics_OL(x_MPC[:,i-1],u_MPC[:,i-1]))
            vector = F @ x_MPC[:,i] - yRCI
            opti.subject_to(vector<=0)
            vector = model.HU @ u_MPC[:,i] - model.hU
            opti.subject_to(vector<=0)
            error = model.output(x_MPC[:,i]) - ref_y[t]
            cost += ca.dot(error,error)

        opti.minimize(cost)
        opti.solver('ipopt',opts)
        sol = opti.solve()
        x_MPC = sol.value(x_MPC).T
        u_MPC = sol.value(u_MPC)
        u_plant[t] = model.u_scaler[0] + (u_MPC[0].T)/model.u_scaler[1]
        parameters_sim[t] = model.parameter(x_model[t], u_MPC[0]).reshape(1,-1)
    else:
        opti = ca.Opti()
        u_MPC = opti.variable(model.nu, N_MPC)
        x_next_1 = model.dynamics(x_model[t], \
                                model.u_scaler[0] + u_MPC[:,0]/model.u_scaler[1], \
                                y_plant[t])
        vector = F @ x_next_1 - yRCI
        opti.subject_to(vector<=0)
        vector = model.HU @ u_MPC[:,0] - model.hU
        opti.subject_to(vector<=0)
        error = model.output(x_next_1) - ref_y[t]
        opti.minimize(ca.dot(error,error))
        opti.solver('ipopt',opts)
        sol = opti.solve()
        u_MPC = sol.value(u_MPC)
        u_plant[t] = model.u_scaler[0] + (u_MPC)/model.u_scaler[1]
        parameters_sim[t] = model.parameter(x_model[t], u_MPC).reshape(1,-1)

    

    #Propagate
    system.update(u_plant[t])
    x_plant[t+1] = system.state
    x_model[t+1] = model.dynamics(x_model[t],u_plant[t],y_plant[t])
    

#Plot results
print_time = np.arange(0, N_sim)
plt.rcParams['text.usetex'] = True
figure, [ax1, ax2, ax3] = plt.subplots(3,1)

ax1.plot(np.arange(0,N_sim)*dt,ref_y,'k--')
ax1.plot(np.arange(0,N_sim)*dt,y_plant, 'g')
ax1.plot(np.arange(0,N_sim)*dt,y_model,'r--')
ax1.plot(np.arange(0,N_sim)*dt,y_max_con*np.ones(N_sim),'k')
ax1.plot(np.arange(0,N_sim)*dt,y_min_con*np.ones(N_sim),'k')
ax1.plot(np.arange(0,N_sim)*dt,y_track_max*np.ones(N_sim),'b:')
ax1.plot(np.arange(0,N_sim)*dt,y_track_min*np.ones(N_sim),'b:')
ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$y$")
ax1.grid(True)
ax1.set_xlim([0, dt*(N_sim-1)]) 

ax2.plot(np.arange(0,N_sim)*dt,u_plant[:,0], 'g')
ax2.plot(np.arange(0,N_sim)*dt,u_bounds_plant[0]*np.ones(N_sim),'k')
ax2.plot(np.arange(0,N_sim)*dt,-u_bounds_plant[0]*np.ones(N_sim),'k')

ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$u$")
ax2.grid(True)
ax2.set_xlim([0, dt*(N_sim-1)]) 

if nx == 2:
    Polytope(A=F, b=yRCI).plot(ax=ax3, patch_args = {"facecolor": 'r', "alpha": 0.8, "linewidth": 1, "linestyle": '-', "edgecolor": 'r'})
else:
    Polytope(A=F, b=yRCI).projection(project_away_dim = np.linspace(2,nx)).plot(ax=ax3, patch_args = {"facecolor": 'r', "alpha": 0.8, "linewidth": 1, "linestyle": '-', "edgecolor": 'k'})

ax3.plot(x_model[:,0], x_model[:,1], 'g', linewidth = 2)
ax3.scatter(x_model[0,0], x_model[0,1], color = 'g')
ax3.autoscale()
ax3.set_xlabel(r"$z_1$")
ax3.set_ylabel(r"$z_2$")
ax3.grid(True)

plt.tight_layout()
plt.show()


