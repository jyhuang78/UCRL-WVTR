import numpy as np
import numpy.linalg as linalg

class RiverSwim:
    """the RiverSwim environment, which is an episodic MDP
    """
    def __init__(self, paras):
        """initialize a riverswim environment

        Args:
            paras: hyperparameter list, which is a dictionary
        """
        self.S = paras['S']
        self.state_space = paras['state_space']
        assert self.S == len(self.state_space)
        self.init_state = paras['init_state']
        self.action_space = paras['action_space']
        self.H = paras['H']

        self.state = self.init_state

    def next_state(self, action):
        """the transition kernel of the episodic MDP RiverSwim, update the state

        Args:
            action: the action chosen by the agent, which the probability of next state depends on
        """
        if self.state == 0:
            if action == 0:
                self.state = 1
            elif action == 1:
                p = [0.1, 0.9]
                self.state = np.random.choice([0,1], p = p)
        elif self.state == self.S-1:
            if action == 0:
                self.state = self.S-2
            elif action == 1:
                p = [0.05, 0.95]
                self.state = np.random.choice([self.S-2, self.S-1], p = p)
        elif 0 < self.state < self.S-1:
            if action == 0:
                self.state -= 1
            elif action == 1:
                p = [0.05, 0.05, 0.9]
                self.state = np.random.choice([self.state-1, self.state, self.state+1], p = p)

    def reset(self):
        """end of episode, set the state to initial state
        """
        self.state = self.init_state

    def reward(self, state, action):
        """specify the reward function of MDP

        Args:
            state: current state
            action: the action chosen by the agent

        Returns:
            the reward function
        """
        if state == 0 and action == 0:
            return 0.005
        elif state == self.S-1 and action == 1:
            return 1
        else:
            return 0

    def phi(self, state, action):
        """specify the features of transition kernel

        Args:
            state: current state
            action: action chosen by the agent

        Returns:
            the feature vector of 'dim' dimesion
        """
        phi_matrix = np.zeros((3, self.S))
        if action == 1:
            if state == 0:
                phi_matrix[:,0] = [0,1,1]
                phi_matrix[:,1] = [1,0,0]
            elif state == self.S-1:
                phi_matrix[:,-2] = [0,1,1]
                phi_matrix[:,-1] = [1,0,0]
            elif 0 < state < self.S-1:
                phi_matrix[:,state-1] = [0,0,1]
                phi_matrix[:,state] = [0,1,0]
                phi_matrix[:,state+1] = [1,0,0]
        elif action == 0:
            phi_matrix[:,max(0,state-1)] = [1,1,1]
        return phi_matrix

class WVTR:
    """the UCRL-WVTR algorithm class
    """
    def __init__(self, paras):
        """initialization, specify the parameters used by UCRL-VTR

        Args:
            paras: the parameters used by UCRL-VTR
        """
        self.i = 0

        self.dim = paras['dim']
        self.M = paras['M']
        self.lam = paras['lam']
        self.state_space = paras['state_space']
        self.action_space = paras['action_space']
        self.H = paras['H']
        self.beta = paras['beta']
        self.sigmamin = paras['sigmamin']
        self.gamma = paras['gamma']

        self.phi_list = []
        self.hat_theta_list = [np.zeros((self.dim, 1))]*self.M
        self.tilde_Sigma_list = [np.identity(self.dim)]*self.M
        self.b_list = [np.zeros((self.dim, 1))]*self.M
        self.hat_Sigma_list = [np.identity(self.dim)]*self.M
        self.v_func = 0

    def action_value_func(self, v_next, reward, phi, state, action):
        """compute the action value function given state-action pair and next-state value functions

        Args:
            v_next: next-state value functions
            reward: reward function
            phi: feature vectors
            state: current state
            action: current action

        Returns:
            Q(s,a) = r(s,a) + [P V](s,a)
        """
        phiv = phi(state, action)@v_next
        theta = self.hat_theta_list[0]
        v_action = reward(state, action) + theta.transpose()@phiv + self.beta*np.sqrt(phiv.transpose()@linalg.solve(self.hat_Sigma_list[0], phiv))
        v_action = np.clip(v_action, 0,1)[0,0]
        return v_action

    def update_v(self, reward, phi):
        """update the value function in a new episode

        Args:
            reward: reward function
            phi: feature vectors
        """
        v_func = np.zeros((len(self.state_space), 1))
        for h in range(self.H, 1, -1):
            v= np.zeros((len(self.state_space),1))
            v_next = v_func[:,0:1]

            for state in self.state_space:
                all_action = []
                for action in self.action_space:
                    v_action = self.action_value_func(v_next, reward, phi, state, action)
                    all_action.append(v_action)
                v[state,0] = max(all_action)
            v_func = np.concatenate((v, v_func), axis = 1)

        self.v_func = v_func

    def update_hat(self):
        """update hat_theta
        """
        self.hat_Sigma_list = self.tilde_Sigma_list.copy()
        self.hat_theta_list = []
        for m in range(self.M):
            self.hat_theta_list.append(linalg.solve(self.hat_Sigma_list[m], self.b_list[m]))

    def select_action(self, state, h, reward, phi):
        """select the action with the maximum value function

        Args:
            state: current state
            h: current step
            reward: reward funtion
            phi: feature vector

        Returns:
            the action with maximum value
        """
        v = self.v_func[:,h:(h+1)]
        all_action = {}
        for i in range(len(self.action_space)):
            action = self.action_space[i]
            all_action[i] = self.action_value_func(v, reward, phi, state, action)

        actions = []

        for i in range(len(self.action_space)):
            if all_action[i] == all_action[max(all_action)]:
                actions.append(i)

        action = np.random.choice(actions)

        return self.action_space[action]

    def update(self,state, action, next_state, h, phi):
        """update historical dataset, estimated variance, Sigma and b

        Args:
            state: current state
            action: action chosen by the agent
            next_state: next state returned by environment
            h: step
            phi: feature vectors
        """
        self.phi_list = []
        for m in range(self.M):
            vm = self.v_func[:,h:(h+1)]**(2**m)
            phiv = phi(state, action)@vm
            self.phi_list.append(phiv)
        bar_sigma = []
        for m in range(self.M-1):
            var_1 = np.clip(self.phi_list[m+1].transpose()@self.hat_theta_list[m+1],0,1)
            var_2 =np.clip(self.phi_list[m].transpose()@self.hat_theta_list[m],0,1)
            variance = var_1 - var_2

            E_1 = np.clip(2*self.beta*np.sqrt(self.phi_list[m].transpose()@linalg.solve(self.hat_Sigma_list[m], self.phi_list[m])),a_min = None, a_max= 1)
            E_2 = np.clip(self.beta*np.sqrt(self.phi_list[m+1].transpose()@linalg.solve(self.hat_Sigma_list[m+1], self.phi_list[m+1])),a_min = None, a_max = 1)
            E = E_1 + E_2

            var_1 = (variance + E)[0,0]
            var_2 = self.sigmamin**2
            var_3 = self.gamma**2*np.sqrt(self.phi_list[m].transpose()@linalg.solve(self.tilde_Sigma_list[m], self.phi_list[m]))

            variance = max(var_1, var_2, var_3)
            bar_sigma.append(variance)

        variance = self.gamma**2*np.sqrt(self.phi_list[-1].transpose()@linalg.solve(self.tilde_Sigma_list[-1], self.phi_list[-1]))
        bar_sigma.append(max(1, self.sigmamin**2, variance))

        for m in range(self.M):
            self.tilde_Sigma_list[m] = self.tilde_Sigma_list[m] + self.phi_list[m]@self.phi_list[m].transpose()/bar_sigma[m]
            self.b_list[m] = self.b_list[m] + self.v_func[next_state,h]**(2**m)*self.phi_list[m]/bar_sigma[m]

def play_wvtr(paras):
    """Solve RiverSwim with UCRL-WVTR
    paras: parameters of RiverSwim and UCRL-WVTR
    K: the number of episodes
    num_path: the number of paths to generate

    save the total regrets within each episode to result (K)
    output results to num_path txt files
    """
    for ii in range(paras['num_path']):
        mdp = RiverSwim(paras)
        algo = WVTR(paras)

        reward_sum = []
        for k in range(paras['K']):
            mdp.reset()

            algo.update_v(mdp.reward, mdp.phi)

            rr = 0
            for h in range(mdp.H):
                state = mdp.state
                action = algo.select_action(mdp.state, h, mdp.reward, mdp.phi)
                rr += mdp.reward(state, action)
                mdp.next_state(action)
                next_state = mdp.state
                algo.update(state, action, next_state, h, mdp.phi)
            algo.update_hat()

            reward_sum.append(rr)
            print('Iteration:', ii, '\tEpisode:', k, '\tReward:', rr)

        file_reward = './riverswim_s_' + str(paras['S']) + '_h_' + str(paras['H']) + '/reward_wvtr_' + str(paras['M']) + '_beta_' + str(paras['beta']) + '_' + str(ii)
        np.savetxt(file_reward, np.array(reward_sum))

def play_random(paras):
    """the policy that chooses actions uniformly from action space

    Args:
        paras: parameters of RiverSwim
    """
    for ii in range(paras['num_path']):
        mdp = RiverSwim(paras)

        reward_sum = []
        for k in range(paras['K']):
            mdp.reset()

            rr = 0
            for h in range(mdp.H):
                state = mdp.state
                action = np.random.choice(mdp.action_space)
                rr += mdp.reward(state, action)
                mdp.next_state(action)

            reward_sum.append(rr)
            print('Iteration:', ii, '\tEpisode:', k, '\tReward:', rr)

        file_reward = './riverswim_s_' + str(paras['S']) + '_h_' + str(paras['H']) + '/reward_random_beta_' + str(paras['beta']) + '_' + str(ii)
        np.savetxt(file_reward, np.array(reward_sum))