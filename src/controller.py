from IPython.display import display
import pandas as pd
from utils import archivo_a_dataframe, preprocessing
from data_processing import df_features, estado_visitas
from environment import ThermalEnv, ACTIONS, n_actions
from controller import q_learning_thermal, comparar_q_tables
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import os

# Crear una carpeta para guardar las Q-tables si no existe
output_dir = "q_tables_output"
import pandas as pd
import ipywidgets as widgets


# Get the unique states from the 'state' column of df_features
unique_states = df_features['state'].unique()

# Initialize the Q-table with zeros for each unique state and action
initial_q_table = {
    state: [0.0 for _ in range(n_actions)]
    for state in unique_states
}

print(initial_q_table)



# Convertir la Q-table a DataFrame
q_table_df = pd.DataFrame.from_dict(initial_q_table, orient='index')
q_table_df.columns = [f'Action {i}' for i in range(n_actions)]

# Mostrar en Colab
display(q_table_df)





rng = np.random.default_rng(seed=42)
env = ThermalEnv()

pi, q_table, q_tables_by_episode, rewards, td_errors_per_episode = q_learning_thermal(
    rng, env,
    n_episodes=10000,
    max_moves=10,
    initial_eps=1.0,
    final_eps=0.05,
    decay_rate=0.99,
    alpha=0.5,
    gamma=0.9,
    q_table=initial_q_table
)

all_actions = env.get_all_actions()

# Filtra solo los estados visitados
estados_visitados = sorted([s for s, c in estado_visitas.items() if c > 0])

# Visualiza
comparar_q_tables(q_tables_by_episode, all_actions)












# Recompensa acumulada
cumulative_rewards = np.cumsum(rewards)

# Recompensa promedio por episodio
average_rewards = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

# Gráfica de recompensa acumulada
plt.figure()
plt.plot(cumulative_rewards)
plt.title("Recompensa Acumulada por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa Acumulada")
plt.grid(True)
plt.show()

# Gráfica de recompensa promedio
plt.figure()
plt.plot(average_rewards)
plt.title("Recompensa Promedio por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa Promedio")
plt.grid(True)
plt.show()












plt.figure()
plt.plot(td_errors_per_episode)
plt.title("Error Temporal Difference por Episodio")
plt.xlabel("Episodio")
plt.ylabel("TD Error promedio")
plt.grid(True)
plt.show()















# Crear un diagrama de cajas del error TD por episodio
plt.figure()
plt.boxplot(td_errors_per_episode)
plt.title("Diagrama de Cajas del Error Temporal Difference por Episodio")
plt.xlabel("Episodio")  # En un boxplot, el eje X generalmente representa grupos o categorías
plt.ylabel("TD Error promedio")
plt.grid(True)
plt.show()



















# Widget para seleccionar el número de episodio
episodio_selector = widgets.IntText(
    value=1,
    description='Episodio:',
    min=1
)

# Botones para mostrar o guardar la Q-table
boton_mostrar = widgets.Button(description="Mostrar Q-table")
boton_guardar = widgets.Button(description="Guardar como CSV")

# Función para mostrar la Q-table
def mostrar_q_table(b):
    episodio = episodio_selector.value
    if 1 <= episodio <= len(q_tables_by_episode):
        q_table = q_tables_by_episode[episodio - 1]
        q_df = pd.DataFrame.from_dict(q_table, orient='index', columns=[f'Action {j}' for j in range(n_actions)])
        display(q_df)
    else:
        print(f"❌ Episodio {episodio} no está en el rango disponible (1–{len(q_tables_by_episode)}).")

# Función para guardar la Q-table como CSV
def guardar_q_table(b):
    episodio = episodio_selector.value
    if 1 <= episodio <= len(q_tables_by_episode):
        q_table = q_tables_by_episode[episodio - 1]
        q_df = pd.DataFrame.from_dict(q_table, orient='index', columns=[f'Action {j}' for j in range(n_actions)])
        filename = f"{output_dir}/q_table_episode_{episodio}.csv"
        q_df.to_csv(filename)
        print(f"✅ Q-table del episodio {episodio} guardada como '{filename}'.")
    else:
        print(f"❌ Episodio {episodio} no está en el rango disponible (1–{len(q_tables_by_episode)}).")

# Asociar funciones a los botones
boton_mostrar.on_click(mostrar_q_table)
boton_guardar.on_click(guardar_q_table)

# Mostrar widgets
display(episodio_selector, boton_mostrar, boton_guardar)

















print("✅ Política aprendida (estado → acción óptima):\n")
for s, a in list(pi.items())[:20]:  # muestra solo los primeros 20
    print(f"{s} → {ACTIONS[a]}")













for s, a in pi.items():
    if s[1] == 0 and s[4] == 0:
        print(f"{s} → Acción aprendida: {ACTIONS[a]}")






# Diccionario de interpretación de variables discretizadas
state_labels = {
    0: {0: "baja (15–23 °C)", 1: "media (24–30 °C)", 2: "alta (31–38 °C)"},  # temp_int
    1: {0: "0 personas", 1: "1–5 personas", 2: "6–10 personas"},              # n_people
    2: {0: "dispersas", 1: "agrupadas cerca ventilación", 2: "agrupadas lejos"},  # location
    3: {0: "muy fría (0–1)", 1: "neutra (2–3)", 2: "muy calurosa (4–5)"},     # opinion
    4: {0: "no hay clase", 1: "hay clase"},                                 # schedule
    5: {0: "baja (20–24 °C)", 1: "media (25–30 °C)", 2: "alta (31–35 °C)"},  # temp_ext
    6: {0: "apagado", 1: "encendido"}                                       # ac_status
}

print("\n🔍 Interpretación detallada de los primeros 20 estados:\n")
for s, a in list(pi.items())[:20]:
    description = []
    for i in range(7):
        val = s[i]
        if isinstance(val, float) and np.isnan(val):
            label = "NaN"
        else:
            label = state_labels[i][val]
        variable_name = [
            "Temp. int", "N° personas", "Ubicación",
            "Opinión térmica", "Horario dentro de 30min", "Temp. ext", "AC"
        ][i]
        description.append(f"{variable_name}: {label}")

    print(" | ".join(description))
    print(f" → Acción recomendada: {ACTIONS[a]}\n")















    # Crear una lista para almacenar las filas del DataFrame
policy_data = []

for s, a in pi.items():
    description = []
    for i in range(7):
        val = s[i]
        variable_name = [
            "temp_interna_discretizada", "n_people", "location",
            "thermal_opinion", "clases a continuación", "temp_externa_discretizada", "estado del aire"
        ][i]
        if i in state_labels and val in state_labels[i]:
             label = state_labels[i][val]
        elif isinstance(val, float) and np.isnan(val):
             label = "NaN"
        else:
             label = str(val) # Fallback to string representation

        description.append(f"{variable_name}: {label}")

    state_description = " | ".join(description)
    optimal_action = ACTIONS[a]

    policy_data.append({
        'State': s,
        'State Description': state_description,
        'Optimal Action': optimal_action
    })

# Crear el DataFrame
policy_df = pd.DataFrame(policy_data)

# Definir la ruta para guardar el CSV
output_dir = "/content/drive/MyDrive/MachineLearning/q_tables_output"
os.makedirs(output_dir, exist_ok=True) # Asegurar que el directorio existe
policy_filename = f"{output_dir}/learned_policy_with_interpretation.csv"

# Guardar el DataFrame en un archivo CSV
policy_df.to_csv(policy_filename, index=False)

print(f"✅ Política aprendida con interpretación guardada en '{policy_filename}'.")