from copy import deepcopy
from collections import defaultdict
import statistics
import numpy as np
import matplotlib.pyplot as plt

estado_visitas = defaultdict(int)

def q_learning_thermal(rng, env, n_episodes=2000, max_moves=10,
                       initial_eps=1.0, final_eps=0.05, decay_rate=0.99,
                       alpha=0.1, gamma=0.9, q_table=None):


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
        td_errors_per_episode.append(sum(total_td_error)/len(total_td_error))

    # Derivar la política final
    pi = {state: np.argmax(q_table[state]) for state in q_table}



    ################## Fuera del aprendizaje ###############################
    plt.figure()
    plt.plot(eps_per_episode)
    plt.title("e - greedy")
    plt.xlabel("Episodio")
    plt.ylabel("EPS")
    plt.grid(True)
    plt.show()


    ################## Fin de la gráfica de eps #######################################################

    return pi, q_table, q_tables_by_episode, rewards_per_episode, td_errors_per_episode