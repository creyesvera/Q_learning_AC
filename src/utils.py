# -*- coding: utf-8 -*-
from IPython.display import display
import ipywidgets as widgets
import pandas as pd
import os
from environment import ACTIONS

def archivo_a_dataframe(ruta_archivo):
  # Verificar que el archivo existe
  if os.path.exists(ruta_archivo):
      # Cargar los datos desde el archivo CSV
      df = pd.read_csv(ruta_archivo)

      # Convertir la columna de fecha a datetime
      df['sensedAt'] = pd.to_datetime(df['sensedAt'])

      # Reorganizar la tabla con pivot
      df_pivot = df.pivot(index='sensedAt', columns='type', values='data')

      # Reiniciar el índice para tener 'sensedAt' como columna
      df_pivot.reset_index(inplace=True)

      # Separar la columna 'sensedAt' en 'Fecha' y 'Hora'
      df_pivot['Fecha'] = df_pivot['sensedAt'].dt.date
      df_pivot['Hora'] = df_pivot['sensedAt'].dt.time

      # Eliminar la columna original 'sensedAt' si ya no es necesaria
      # df_pivot.drop('sensedAt', axis=1, inplace=True) # Uncomment if you want to remove the original column

      # Mostrar la tabla
      print("Tabla de datos reorganizada:")
      print(df_pivot)
  else:
      print(f"No se encontró el archivo en la ruta especificada: {ruta_archivo}")
  return df_pivot











def graficar_datos(df_pivot):

  # Obtener columnas de datos disponibles (excluyendo 'sensedAt', 'Fecha', 'Hora')
  value_columns = df_pivot.columns.tolist()
  columns_to_remove = ['sensedAt', 'Fecha', 'Hora']
  value_columns = [col for col in value_columns if col not in columns_to_remove]

  if not value_columns:
      print("No hay columnas numéricas para graficar.")
      return

  # Crear widgets
  value_dropdown = widgets.Dropdown(
      options=value_columns,
      description='Seleccione el valor:',
      disabled=False,
  )

  plot_button = widgets.Button(
      description='Mostrar Gráfica',
      disabled=False,
      button_style='success',
      tooltip='Mostrar gráfica',
      icon='bar-chart'
  )

  output_widget = widgets.Output() # Widget para mostrar la gráfica y mensajes

  # Función para manejar el clic del botón
  def on_plot_button_clicked(b):
      selected_value = value_dropdown.value

      if selected_value in df_pivot.columns:
          plt.figure(figsize=(12, 6))
          plt.plot(df_pivot['sensedAt'], df_pivot[selected_value])
          plt.xlabel('Tiempo (sensedAt)')
          plt.ylabel(selected_value)
          plt.title(f'Gráfica de {selected_value} vs Tiempo')
          plt.grid(True)

          plt.xticks(
              ticks=df_pivot['sensedAt'][::50],
              labels=df_pivot['Hora'][::50],
              rotation=90
          )

          plt.tight_layout()
          plt.show()
      else:
          print(f"La columna '{selected_value}' no se encontró en el DataFrame.")

  # Asignar la función al evento de clic del botón
  plot_button.on_click(on_plot_button_clicked)

  # Mostrar los widgets
  print("Seleccione el valor que desea graficar:")
  display(value_dropdown, plot_button)


















def preprocessing(df, columnas_a_filtrar, limites_outliers):
    # Asegurar que sensedAt sea datetime
    df['sensedAt'] = pd.to_datetime(df['sensedAt'])
    df.sort_values('sensedAt', inplace=True)
    df.set_index('sensedAt', inplace=True)

    # Convertir objetos a tipos inferidos antes de interpolar (evita FutureWarning)
    df = df.infer_objects(copy=False)

    # Interpolación temporal
    df_interpolated = df.interpolate(method='time')

    # Reemplazo de NaNs en los bordes
    df_interpolated = df_interpolated.bfill()
    df_interpolated = df_interpolated.ffill()

    # Filtro por límites (outliers)
    for col, (min_val, max_val) in limites_outliers.items():
        if col in df_interpolated.columns:
            df_interpolated = df_interpolated[df_interpolated[col].between(min_val, max_val)]

    return df_interpolated
















































import matplotlib.pyplot as plt
import numpy as np

def comparar_q_tables(q_tables_by_episode, all_actions):
    diferencias = []

    for i in range(1, len(q_tables_by_episode)):
        q_anterior = q_tables_by_episode[i - 1]
        q_actual = q_tables_by_episode[i]

        diff_total = 0.0
        for estado in set(q_anterior.keys()).union(q_actual.keys()):
            q1 = np.array(q_anterior.get(estado, [0.0] * len(all_actions)))
            q2 = np.array(q_actual.get(estado, [0.0] * len(all_actions)))
            diff_total += np.sum(np.abs(q2 - q1))

        diferencias.append(diff_total)

    # Graficar
    plt.figure(figsize=(8, 4))
    plt.plot(diferencias, color='darkblue')
    plt.title('Evolución de la diferencia total en Q-table por episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Suma de |ΔQ(s,a)|')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #return diferencias
    return




import numpy as np
import matplotlib.pyplot as plt

def comparar_q_tables_heatmap(q_tables_by_episode, estados_visitados, all_actions):
    diffs_por_par = []  # Lista de listas: cada sublista tiene ΔQ(s,a) para un episodio

    for i in range(1, len(q_tables_by_episode)):
        q_anterior = q_tables_by_episode[i - 1]
        q_actual = q_tables_by_episode[i]

        diffs_estado_accion = []
        for s in estados_visitados:
            for a_idx, a in enumerate(all_actions):
                q1 = q_anterior.get(s, [0.0] * len(all_actions))[a_idx]
                q2 = q_actual.get(s, [0.0] * len(all_actions))[a_idx]
                delta = abs(q2 - q1)
                diffs_estado_accion.append(delta)

        diffs_por_par.append(diffs_estado_accion)

    diffs_matrix = np.array(diffs_por_par)  # shape: [episodios-1, estados*acciones]

    # Heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(diffs_matrix.T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='|ΔQ(s,a)|')
    plt.xlabel('Episodio')
    plt.ylabel('Índice (s,a)')
    plt.title('Evolución de ΔQ(s,a) en cada episodio')
    plt.tight_layout()
    plt.show()

    return diffs_matrix



















