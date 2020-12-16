import numpy as np
from datetime import datetime as dt

def main():
    niter = 100
    step_size = 0.5
    
    seed_num = int(dt.strftime(dt.now(), '%H%f'))
    np.random.seed(seed_num)
    
    # initial 
    x = 0
    naccept = 0
    
    # sampling
    for i in range(1, niter):
        backup_x = x
        action_init = 0.5 * x * x
        
        dx = np.random.rand()
        dx = (dx - 0.5) * step_size * 2
        x = x + dx
        
        action_fin = 0.5*x*x
        
        metropolis = np.random.rand()
        if np.exp(action_init - action_fin) > metropolis:
            naccept += 1
        else:
            x = backup_x
        print(x, ' : ', naccept/niter)
        
if __name__ == '__main__':
    main()