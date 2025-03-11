"""
Initialization for qLPV+RCI identification
"""

from imports import *
from utils.qLPV_BFR import qLPV_BFR

def initialization(dataset, sizes, kappa, id_params, activation_func = 1):
    
    if activation_func == 1:
        activation = nn.swish
    elif activation_func == 2:
        activation = nn.relu
    elif activation_func == 3:
        @jax.jit
        def activation(x):
            return nn.elu(x)+1.
    elif activation_func == 4:
        activation = nn.sigmoid

    #Extract data
    Y_train = dataset['Y_train']
    Y_test = dataset['Y_test']
    Ys_train = dataset['Ys_train']
    Us_train = dataset['Us_train']
    Ys_observer = dataset['Ys_observer']
    Us_observer = dataset['Us_observer']
    Ys_test = dataset['Ys_test']
    Us_test = dataset['Us_test']
    ny = Ys_train[0].shape[1]
    nu = Us_train[0].shape[1]
    nx = sizes[0]
    nq = sizes[1]
    nth = sizes[2]
    nH = sizes[3]
    constraints = dataset['constraints']    

    # Optimization params
    iprint = id_params['iprint']
    memory = id_params['memory']
    eta = id_params['eta']
    rho_th = id_params['rho_th']
    adam_epochs = id_params['adam_epochs']
    lbfgs_epochs = id_params['lbfgs_epochs']
    train_x0 = id_params['train_x0']
    weight_RCI = id_params['weight_RCI']
    N_MPC = id_params['N_MPC']

    jax.config.update("jax_enable_x64", True)

    #Extract sizes
    print('Identifying model with nx:', nx, 'nq:', nq, 'nth:', nth, 'nH:', nH)

    from LTI_identification import initial_LTI_identification  
    model_LTI, RCI_LTI = initial_LTI_identification(Ys_train, Us_train, Ys_observer, Us_observer, nx, constraints, kappa, N_MPC)

    #Do system identification inside the RCI set
    #Extract LTI data
    A_LTI = model_LTI['A']
    B_LTI = model_LTI['B']
    C_LTI = model_LTI['C']

    # Define the optimization variables
    key = jax.random.PRNGKey(10)
    key1, key2, key3 = jax.random.split(key, num=3)
    A = 0.0001*jax.random.normal(key1, (nq,nx,nx))
    B = 0.0001*jax.random.normal(key2, (nq,nx,nu))
    C = 0.0001*jax.random.normal(key2, (ny,nx))
    Win = 0.0001*jax.random.normal(key1, (nq, nth, nx))
    bin = 0.0001*jax.random.normal(key2, (nq, nth))
    Whid = 0.0001*jax.random.normal(key3, (nq, nH-1, nth, nth))
    bhid = 0.0001*jax.random.normal(key1, (nq, nH-1, nth))
    Wout = 0.0001*jax.random.normal(key2, (nq, nth))
    bout = 0.0001*jax.random.normal(key3, (nq, ))

    C = jnp.array(C_LTI)
    for i in range(nq):
        A = A.at[i].set(jnp.array(A_LTI.copy()))
        B = B.at[i].set(jnp.array(B_LTI.copy()))
       
    A_new = np.array(A)
    B_new = np.array(B)
    C_new = np.array(C)
    Win_new = np.array(Win)
    bin_new = np.array(bin)
    Whid_new = np.array(Whid)
    bhid_new = np.array(bhid)
    Wout_new = np.array(Wout)
    bout_new = np.array(bout)
    model_LPV = {'A': A_new, 'B': B_new, 'C': C_new, 'Win': Win_new, 'bin': bin_new, 'Whid': Whid_new, 'bhid': bhid_new, 'Wout': Wout_new, 'bout': bout_new}
    model_LPV['L'] = np.zeros((nq, nx, ny))

    #Check BFRs
    model_LTI_BFR = {'A': A, 'B': B, 'C': C, 'Win': Win, 'bin': bin, 'Whid': Whid, 'bhid': bhid, 'Wout': Wout, 'bout': bout}
    BFR_train_qLPV, y_train_qLPV, x0s_train_qLPV = qLPV_BFR(model_LPV, Us_train, Ys_train, observer = False, activation_func = activation_func)
    print('BFR train: qLPV', BFR_train_qLPV)
    BFR_observer_qLPV, y_observer_qLPV, x0s_observer_qLPV = qLPV_BFR(model_LPV, [Us_observer], [Ys_observer], observer = False, activation_func = activation_func)
    print('BFR observer: qLPV', BFR_observer_qLPV)
    BFR_test_qLPV, y_test_qLPV, x0s_test_qLPV = qLPV_BFR(model_LPV, [Us_test], [Ys_test], observer = False, activation_func = activation_func)
    print('BFR test: qLPV', BFR_test_qLPV)

    #Save sim data
    model_LPV['yhat_train'] = y_train_qLPV
    model_LPV['yhat_observer'] = np.array(y_observer_qLPV[0])
    model_LPV['yhat_test'] = np.array(y_test_qLPV[0])

    return model_LPV, RCI_LTI



        



