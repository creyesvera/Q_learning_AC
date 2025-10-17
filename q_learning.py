import random
from itertools import product
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


ACTIONS = {
    0: "Prender ventilación",
    1: "Bajar temperatura",
    2: "No hacer nada",
    3: "Subir temperatura",
    4: "Apagar ventilación"
}



class ThermalEnv:
    def __init__(self, reward_table=None):
        self.state_space = list(product(range(3),  # temp_int
                                        range(3),  # n_people
                                        range(3),  # location
                                        range(3),  # thermal_opinion
                                        range(2),  # schedule
                                        range(3),  # temp_ext
                                        range(2))) # ac_status
        self.actions = [0, 1, 2, 3, 4]
        self.valid_states = [s for s in self.state_space if self.is_valid_state(s)]
        self.current_state = None
        self.reward_table = reward_table # Add reward_table as an attribute

    def is_valid_state(self, state):
        temp_int, n_people, location, opinion, schedule, temp_ext, ac_status = state

        # Regla 1: Sin personas → no hay ubicación ni opinión
        if n_people == 0:
            if location != 0:
                return False
            if opinion != 1:  # Neutral
                return False

        # Regla 2: Frío interno y externo → no puede sentir calor
        if temp_int == 0 and temp_ext == 0 and opinion == 2:
            return False

        # Regla 3: Calor interno y externo → no puede sentir frío
        if temp_int == 2 and temp_ext == 2 and opinion == 0:
            return False

        return True



    def reset(self):
        # 30% de las veces, elige un estado sin personas ni clases
        if random.random() < 0.3:
            candidatos = [s for s in self.valid_states if s[1] == 0 and s[4] == 0]
            if candidatos:
                self.current_state = random.choice(candidatos)
                return self.current_state

        # En los otros casos, elige un estado aleatorio normal
        self.current_state = random.choice(self.valid_states)
        return self.current_state


    def get_actions(self, state):
        temp_int, n_people, location, opinion, schedule, temp_ext, ac_status = state
        if n_people == 0 and schedule == 0:
            return [4] if ac_status == 1 else [2]
        return self.actions.copy()

    def step(self, state, action):
        temp_int, n_people, location, opinion, schedule, temp_ext, ac_status = state

        if action == 0:
            ac_status = 1
        elif action == 1:
            temp_int = max(0, temp_int - 1)
        elif action == 3:
            temp_int = min(2, temp_int + 1)
        elif action == 4:
            ac_status = 0

        # The transition logic for opinion should ideally be based on the action and current state.
        # For now, keeping the random transition as in the original code, but note this is a simplification.
        if temp_int == 0 and temp_ext == 0:
            opinion = random.choice([0, 1])
        elif temp_int == 2 and temp_ext == 2:
            opinion = random.choice([1, 2])
        else:
            opinion = random.randint(0, 2)

        next_state = (temp_int, n_people, location, opinion, schedule, temp_ext, ac_status)
        # Ensure the next state is valid after an action
        # This loop can potentially run infinitely if no valid state is reachable.
        # A more robust approach might be to redefine transitions or handle invalid states differently.
        attempts = 0
        while not self.is_valid_state(next_state) and attempts < 10:
            opinion = random.randint(0, 2)
            next_state = (temp_int, n_people, location, opinion, schedule, temp_ext, ac_status)
            attempts += 1

        # Forzar a un estado válido si fue imposible encontrar uno por azar
        if not self.is_valid_state(next_state):
            if n_people == 0:
                location = 0
                opinion = 1
            elif temp_int == 0 and temp_ext == 0:
                opinion = random.choice([0, 1])  # solo frío o neutral
            elif temp_int == 2 and temp_ext == 2:
                opinion = random.choice([1, 2])  # solo calor o neutral
            else:
                opinion = 1  # neutro por defecto

            next_state = (temp_int, n_people, location, opinion, schedule, temp_ext, ac_status)

        reward = self.compute_reward(state, action, next_state)
        return next_state, reward


    def compute_reward(self, state, action, next_state):
        # Use the provided reward table if it exists
        if self.reward_table is not None:
            # Assuming the reward table is a dictionary mapping (state, action, next_state) to reward
            # Or a simpler structure if the reward only depends on (state, action) or next_state
            # For this example, let's assume the reward is based on the next_state and action
            # A more complex table could be used if needed.
            try:
                # Example: If reward_table is a dict mapping (next_state, action) to reward
                return self.reward_table.get((next_state, action), 0) # Default to 0 if not in table
            except TypeError:
                # Handle cases where reward might only depend on next_state if table structure is different
                try:
                    return self.reward_table.get(next_state, 0)
                except:
                    # Fallback or error handling if the table structure is unexpected
                    print("Warning: Could not retrieve reward from provided table. Using default.")
                    pass # Fallback to simplified reward if table lookup fails


        # Simplified reward function based on thermal opinion (used if reward_table is None or lookup fails)
        temp_int, n_people, location, opinion, schedule, temp_ext, ac_status = next_state

        # ✅ Regla 0: No se puede subir o bajar la temperatura si el aire no está
        # encendido primero o apagar si ya está apagado o prender si ya está prendido
        if (ac_status == 0 and action in [1, 3, 4]) or (ac_status == 1 and action == 0):
            #print("Regla 0, reward: -2")
            return -2

        # ✅ Regla 1: No hay personas ni clases → apagar o mantener apagado el AC
        if n_people == 0 and schedule == 0:
            if (ac_status == 1 and action == 4) or (ac_status == 0 and action == 2):
                #print("Regla 1, reward: 2")
                return 1
            else:
                #print("Regla 1, reward: -2")
                return -1

        # ✅ Regla 2: Si en 30 minutos hay clase y hace calor → El aire debe estar
        # encendido, manternerse encendido o bajar la temperatura
        if schedule == 1 and temp_int == 2:
            if (ac_status == 1 and action in [1, 2]) or (ac_status == 0 and action == 0):
                if temp_ext == 2:
                #print("Regla 2, reward: 2")
                    return 1
                #print("Regla 2, reward: 1")
            return 1

        # ✅ Regla 3: Está muy caliente, temperatura alta  → El aire debe estar
        # encendido, manternerse encendido o bajar la temperatura
        if opinion == 2 and temp_int == 2:
            if (ac_status == 1 and action in [1, 2]) or (ac_status == 0 and action == 0):
                if temp_ext == 2:
                #print("Regla 3 reward: 2")
                    return 1
            #print("Regla 3, reward: 1")
            return 1


        # ✅ Regla 4: Está muy frío, temperatura baja  → El aire debe estar
        # apagado, manternerse apagado o subir la temperatura
        if opinion == 0 and temp_int == 0:
            if (ac_status == 1 and  action in [3, 4]) or (ac_status == 0 and action == 2):
                if temp_ext == 0:
                #print("Regla 4, reward: 2")
                    return 1
            #print("Regla 4, reward: 1")
            return 1


        # ✅ Regla 5: Opinión térmica es cómoda
        if opinion == 1: # Neutral/Comfortable
            #print("Regla 5, reward: 1")
            return 1
        elif opinion == 0: # Too Cold
            #print("Regla 5, reward: 0")
            return 0
        elif opinion == 2: # Too Hot
            #print("Regla 5, reward: -1")
            return -1

        return 0


    def get_all_states(self):
        return self.valid_states

    def get_all_actions(self):
        return list(ACTIONS.keys())
    
    


from copy import deepcopy
from collections import defaultdict
import statistics

estado_visitas = defaultdict(int)

def q_learning_thermal(rng, env, n_episodes=2000, max_moves=10,
                        initial_eps=1.0, final_eps=0.05, decay_rate=0.99,
                        alpha=0.5, gamma=0.9, q_table=None):


    if q_table is None:
        q_table = defaultdict(lambda: np.zeros(len(env.get_all_actions())))
    else:
        q_table = defaultdict(lambda: np.zeros(len(env.get_all_actions())), q_table)
        print("Se usa la tabla ingresada en el parametro")

    q_tables_by_episode = []
    rewards_per_episode = []
    td_errors_per_episode = []
    eps_per_episode = []

    for epi in range(n_episodes):
        total_reward = 0
        total_td_error = []
        #eps debe ir cayendo
        eps = max(final_eps, initial_eps * (decay_rate ** epi)) #print a la parte de explorar y explotar y un contador
        eps_per_episode.append(eps)
        #explorar bastante y explotar lo que ya se tiene grabado
        #estudiar bien esta función max(final_eps, initial_eps * (decay_rate ** epi))
        #10000 episodios para ver que sale

        ##### Inicializar S ##############################
        state = env.reset()

        ##### Repetir para cada paso del episodio ##############################
        for _ in range(max_moves): #Pasos del episodio si nada sale bien poner 20

        ##### Escoger A de S usando la política e-greedy ##############################
            actions = env.get_actions(state)
            if rng.random() < eps: #explorar
                action = rng.choice(actions)
            else: #explotar
                q_values = q_table[state]
                action = actions[np.argmax([q_values[a] for a in actions])]

        ##### Tomar una acción A y observar S' y R (estado siguiente y recompensa) ##############################
            next_state, reward = env.step(state, action)
            total_reward += reward


        ##### Actualizar tabla Q ##############################
            # Q-learning update
            next_actions = env.get_actions(next_state) #
            max_next_q = max([q_table[next_state][a] for a in next_actions]) if next_actions else 0

            td = reward + gamma * max_next_q - q_table[state][action]
            #print(td)

            #SARSA: reward + gamma * next_q - q_table[state][action]
            q_table[state][action] += alpha * td
        ##### S <- S' ##############################
            state = next_state
            total_td_error.append(abs(td))


        # Guardar copia para analizar evolución
        q_tables_by_episode.append(q_table.copy())
        rewards_per_episode.append(total_reward)
        td_errors_per_episode.append(statistics.mean(total_td_error))

    # Derivar la política final
    pi = {state: np.argmax(q_table[state]) for state in q_table}



    ################## Fuera del aprendizaje (luego borrar este bloque) ###############################
    plt.figure()
    plt.plot(eps_per_episode)
    plt.title("e - greedy")
    plt.xlabel("Episodio")
    plt.ylabel("EPS")
    plt.grid(True)
    plt.show()


    ################## Fin de la gráfica de eps #######################################################

    return pi, q_table, q_tables_by_episode, rewards_per_episode, td_errors_per_episode
