from utils import archivo_a_dataframe, preprocessing
import pandas as pd
import numpy as np

## Datos de labclimate y la estación meteorológica
labclimateUno = "data\hayIot-labclimateUno.csv"
labclimateTres = "data\hayIot-labclimateTres.csv"
labclimateDos = "data\hayIot-labclimateDos.csv"
labclimateCuatro = "data\hayIot-labclimateCuatro.csv"
estMeteorologica = "data\hayIot-Estación Meteorológica Jardín Delantero LST.csv"



## Datos del aire acondicionado
Shelly_Iz = "data\hayIot-Shelly_Iz.csv"       ##Solo un shelly porque el otro está dañado





horario = {
    'Hora': [
        '7:00-7:30', '7:30-8:00', '8:00-8:30', '8:30-9:00', '9:00-9:30',
        '9:30-10:00', '10:00-10:30', '10:30-11:00', '11:00-11:30', '11:30-12:00',
        '12:00-12:30', '12:30-13:00', '13:00-13:30', '13:30-14:00', '14:00-14:30',
        '14:30-15:00', '15:00-15:30', '15:30-16:00', '16:00-16:30', '16:30-17:00',
        '17:00-17:30', '17:30-18:00', '18:00-18:30'
    ],
    'Lunes': [
        'Vacío', 'Clases', 'Clases', 'Clases', 'Clases', 'Clases', 'Clases', 'Clases',
        'Clases', 'Clases', 'Clases', 'Clases', 'Vacío', 'Clases', 'Clases', 'Clases',
        'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Vacío'
    ],
    'Martes': [
        'Vacío', 'Laboratorio abierto', 'Laboratorio abierto', 'Laboratorio abierto', 'Clases',
        'Clases', 'Clases', 'Clases', 'Clases', 'Clases', 'Clases', 'Clases', 'Vacío',
        'Vacío', 'Laboratorio abierto', 'Laboratorio abierto', 'Laboratorio abierto',
        'Laboratorio abierto', 'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Vacío'
    ],
    'Miércoles': [
        'Vacío', 'Laboratorio abierto', 'Laboratorio abierto', 'Laboratorio abierto',
        'Laboratorio abierto', 'Clases', 'Clases', 'Clases', 'Clases', 'Clases',
        'Clases', 'Clases', 'Clases', 'Clases', 'Clases', 'Vacío', 'Vacío', 'Vacío',
        'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Vacío'
    ],
    'Jueves': [
        'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Laboratorio abierto', 'Laboratorio abierto',
        'Laboratorio abierto', 'Laboratorio abierto', 'Laboratorio abierto', 'Clases',
        'Clases', 'Clases', 'Clases', 'Clases', 'Laboratorio abierto', 'Laboratorio abierto',
        'Laboratorio abierto', 'Laboratorio abierto', 'Vacío', 'Vacío', 'Vacío', 'Vacío',
        'Vacío'
    ],
    'Viernes': [
        'Vacío', 'Vacío', 'Vacío', 'Vacío', 'Clases', 'Clases', 'Clases', 'Clases',
        'Clases', 'Clases', 'Clases', 'Clases', 'Vacío', 'Vacío', 'Laboratorio abierto',
        'Laboratorio abierto', 'Laboratorio abierto', 'Vacío', 'Vacío', 'Vacío', 'Vacío',
        'Vacío', 'Vacío'
    ]
}





























df_labclimateUno = archivo_a_dataframe(labclimateUno)
df_labclimateTres = archivo_a_dataframe(labclimateTres)
df_labclimateDos = archivo_a_dataframe(labclimateDos)
df_labclimateCuatro = archivo_a_dataframe(labclimateCuatro)
df_estMeteorologica = archivo_a_dataframe(estMeteorologica)

df_shelly_iz = archivo_a_dataframe(Shelly_Iz)
















columnas_a_conservar = ['Fecha', 'Hora', 'sensedAt', 'temp']


df_labclimateUno = df_labclimateUno[columnas_a_conservar]
print(df_labclimateUno.head())


df_labclimateTres = df_labclimateTres[columnas_a_conservar]
print(df_labclimateTres.head())


df_labclimateDos = df_labclimateDos[columnas_a_conservar]
print(df_labclimateDos.head())


df_labclimateCuatro = df_labclimateCuatro[columnas_a_conservar]
print(df_labclimateCuatro.head())


df_estMeteorologica = df_estMeteorologica[columnas_a_conservar]
print(df_estMeteorologica.head())



















columnas_a_conservar = ['sensedAt','Fecha', 'Hora', 'potencia_A']
df_shelly_iz = df_shelly_iz[columnas_a_conservar]
print(df_shelly_iz.head())














# Crear el DataFrame de pandas
df_horario = pd.DataFrame(horario)

# Función para determinar si hay clases o laboratorio abierto
def get_activity_type(cell_content):
    if "Laboratorio abierto" in str(cell_content):
        return "Clases"
    elif str(cell_content).strip() and str(cell_content) != 'Vacío':
        return "Clases"
    else:
        return "No hay clases"

# Aplicar la función a cada celda de los días de la semana
for day in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']:
    df_horario[day] = df_horario[day].apply(get_activity_type)

# Imprimir el DataFrame
print(df_horario)



















tiempo = 30 #min













columnas = ['temp']
limites = {
    'temp': (15, 38),
}

df_scaled_labclimateUno = preprocessing(df_labclimateUno, columnas_a_filtrar=columnas, limites_outliers=limites)
df_scaled_labclimateTres = preprocessing(df_labclimateTres, columnas_a_filtrar=columnas, limites_outliers=limites)
df_scaled_labclimateDos = preprocessing(df_labclimateDos, columnas_a_filtrar=columnas, limites_outliers=limites)
df_scaled_labclimateCuatro = preprocessing(df_labclimateCuatro, columnas_a_filtrar=columnas, limites_outliers=limites)











# Combinar los 4 DataFrames
df_combinado = pd.concat([
    df_scaled_labclimateUno.reset_index(),
    df_scaled_labclimateTres.reset_index(),
    df_scaled_labclimateDos.reset_index(),
    df_scaled_labclimateCuatro.reset_index()
])

# Agrupar por sensedAt, Fecha y Hora, y calcular el promedio de 'temp'
df_temperatura_interna = df_combinado.groupby(['sensedAt', 'Fecha', 'Hora']).agg(temp=('temp', 'mean')).reset_index()

# Mostrar el resultado
print(df_temperatura_interna.head())






















print(df_temperatura_interna[df_temperatura_interna['temp'].isna()])


















# Discretizar la temperatura interna
bins = [15, 23, 30, 38]
labels = [0, 1, 2]
df_temperatura_interna['temp_discretizada'] = pd.cut(df_temperatura_interna['temp'], bins=bins, labels=labels, right=False)

print(df_temperatura_interna.head())













columnas = ['temp']
limites = {
    'temp': (15, 40),
}

df_scaled_estMeteorologica = preprocessing(df_estMeteorologica, columnas_a_filtrar=columnas, limites_outliers=limites)












# Discretizar la temperatura externa
bins = [20, 24, 30, 35]
labels = [0, 1, 2]
df_scaled_estMeteorologica['temp_discretizada'] = pd.cut(df_scaled_estMeteorologica['temp'], bins=bins, labels=labels, right=False)

print(df_scaled_estMeteorologica.head())

















df_shelly_iz['estado del aire'] = (df_shelly_iz['potencia_A'] > 20).astype(int)
print(df_shelly_iz.head())


















print(df_shelly_iz[df_shelly_iz['estado del aire'] == 1])
















# Merge progresivo
merged_df = pd.merge(df_temperatura_interna, df_scaled_estMeteorologica.reset_index(), on=['Fecha', 'Hora'], how='outer')
merged_df = pd.merge(merged_df, df_shelly_iz, on=['Fecha', 'Hora'], how='outer')

# Crear columna sensedAt tomando la primera no nula
merged_df['sensedAt_final'] = merged_df['sensedAt_x'].combine_first(
    merged_df['sensedAt_y'].combine_first(
        merged_df['sensedAt']
    )
)

# Renombrar y reorganizar
df_features = merged_df[[
    'sensedAt_final', 'Fecha', 'Hora',
    'temp_discretizada_x', 'temp_discretizada_y',
    'potencia_A', 'estado del aire'
]].rename(columns={
    'sensedAt_final': 'sensedAt',
    'temp_discretizada_x': 'temp_interna_discretizada',
    'temp_discretizada_y': 'temp_externa_discretizada'
})


# Mostrar filas con NaN
print("Filas con valores NaN:")
print(df_features[df_features.isna().any(axis=1)])

















# Limpieza inicial
df_horario['Hora'] = df_horario['Hora'].str.strip()

# Expandimos el rango de horas en una estructura más manejable
def expand_range(row):
    start, end = row['Hora'].split('-')
    return pd.Series({
        'start_time': pd.to_datetime(start, format='%H:%M').time(),
        'end_time': pd.to_datetime(end, format='%H:%M').time()
    })

df_horario[['start_time', 'end_time']] = df_horario.apply(expand_range, axis=1)


















# Añadir la columna "clases a continuación"
df_features['clases a continuación'] = 0  # Falso

# Convertir la columna 'Hora' de df_features a objetos time para comparación
df_features['Hora_time'] = df_features['Hora'].apply(lambda x: pd.to_datetime(str(x)).time())

# Convertir la columna 'Fecha' de df_features a datetime para obtener el día de la semana
df_features['Fecha_datetime'] = pd.to_datetime(df_features['Fecha'])

# Iterar sobre cada fila de df_features
for index, row in df_features.iterrows():
    current_date = row['Fecha_datetime']
    current_time = row['Hora_time']

    # Obtener el nombre del día de la semana en español
    day_of_week = current_date.strftime('%A')
    # Mapear a los nombres de columnas en df_horario
    day_mapping = {
        'Monday': 'Lunes',
        'Tuesday': 'Martes',
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': None,  # No hay clases los sábados según el horario
        'Sunday': None     # No hay clases los domingos según el horario
    }
    day_column = day_mapping.get(day_of_week)

    if day_column is not None:
        # Calcular el tiempo 30 minutos en el futuro
        current_datetime = pd.to_datetime(f'{current_date.strftime("%Y-%m-%d")} {current_time.strftime("%H:%M:%S")}')
        future_datetime = current_datetime + pd.Timedelta(minutes=tiempo)
        future_time = future_datetime.time()

        # Verificar si hay alguna entrada en df_horario en el rango [current_time, future_time]
        # donde la actividad para el día de la semana sea 'Clases'
        for hor_index, hor_row in df_horario.iterrows():
            hor_start_time = hor_row['start_time']
            hor_end_time = hor_row['end_time']
            activity = hor_row[day_column]

            # Convertir los objetos time a valores comparables como strings
            # Esto es para manejar el caso en que future_time cruce la medianoche, aunque
            # con un rango de 30 minutos es poco probable dentro del horario dado.
            # Una comparación más robusta implicaría convertir a segundos desde medianoche.
            # Sin embargo, para este rango y horario, una comparación simple de objetos time es suficiente.

            # Verificar si el intervalo [current_time, future_time] se superpone con [hor_start_time, hor_end_time)
            # Comprobamos la superposición considerando la ventana de 30 minutos.
            # Una clase está "a continuación" si alguna parte de los próximos 30 minutos cae dentro de un bloque de clase.

            # Simplificación: verificar si la hora actual o cualquier hora dentro de los próximos 30 minutos cae en un bloque de clase.
            # Esto es equivalente a comprobar si los intervalos se superponen.

            # La superposición existe si (inicio1 < fin2) y (inicio2 < fin1)
            # Aquí: inicio1 = current_time, fin1 = future_time
            #       inicio2 = hor_start_time, fin2 = hor_end_time

            # Convertir horas a valores numéricos (por ejemplo, minutos desde medianoche)
            current_minutes = current_time.hour * 60 + current_time.minute
            future_minutes = future_time.hour * 60 + future_time.minute
            hor_start_minutes = hor_start_time.hour * 60 + hor_start_time.minute
            hor_end_minutes = hor_end_time.hour * 60 + hor_end_time.minute

            # Manejar cruce de medianoche si future_time cruza las 00:00 (poco probable con solo 30 min)
            if future_minutes < current_minutes:  # Cruzó medianoche
                future_minutes += 24 * 60

            # Manejar caso especial cuando end_time es 00:00 (se interpreta como fin del día)
            # En este horario específico, los rangos están dentro de un solo día, así que no es estrictamente necesario.
            # Pero para mayor robustez, si hor_end_minutes es 0, considerarlo como fin de día.

            # Verificar superposición de los intervalos
            # Nota: En definiciones típicas, el final del intervalo es exclusivo,
            # pero el horario usa bloques fijos de 30 minutos. Supondremos [inicio, fin] por simplicidad,
            # lo que implica que si una clase termina a las 8:00, cubre hasta las 8:00.
            # Una suposición segura para "a continuación" es si el bloque de 30 min se cruza con cualquier clase.

            # Verificar si una clase inicia dentro de los próximos 30 minutos,
            # o si la hora actual ya está dentro de una clase que se extiende más allá.

            # Escenario 1: Una clase empieza dentro de los próximos 30 minutos.
            if current_minutes <= hor_start_minutes < future_minutes:
                if activity == 'Clases':
                    df_features.at[index, 'clases a continuación'] = 1  # Verdadero
                    break  # No es necesario verificar otras filas del horario

            # Escenario 2: La hora actual está dentro de una clase que continúa durante los próximos 30 minutos.
            if hor_start_minutes <= current_minutes < hor_end_minutes:
                if activity == 'Clases':
                    df_features.at[index, 'clases a continuación'] = 1  # Verdadero
                    break  # No es necesario verificar otras filas del horario

# Eliminar las columnas temporales si ya no se necesitan
df_features.drop(columns=['Hora_time', 'Fecha_datetime'], inplace=True)

# Mostrar las primeras filas del DataFrame con la nueva columna
print("\nDataFrame df_features con la nueva columna 'clases a continuación':")
print(df_features.head())

# Opcional: Mostrar algunas filas donde 'clases a continuación' sea 1 para verificación
print("\nFilas donde 'clases a continuación' es 1:")
print(df_features[df_features['clases a continuación'] == 1].head())




























np.random.seed(42)

df_features['n_people'] = np.random.choice([0, 1, 2], size=len(df_features), p=[0.3, 0.5, 0.2])
df_features['location'] = np.random.choice([0, 1, 2], size=len(df_features), p=[0.5, 0.25, 0.25])
df_features['thermal_opinion'] = np.random.choice([0, 1, 2], size=len(df_features), p=[0.2, 0.6, 0.2])