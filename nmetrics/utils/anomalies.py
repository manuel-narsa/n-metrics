import numpy as np

def escaner_anomalias_ultra_rapido(dict_estados, m_jueces, topologia):
    """
    Escáner de anomalías en O(U) donde U son las Clases de Equivalencia únicas,
    ignorando por completo la N real de la matriz (Big Data).
    """
    # Array para acumular la "penalización" o "desviación" de cada juez
    penalizaciones_jueces = np.zeros(m_jueces)
    total_votos_juez = np.zeros(m_jueces)
    
    # Iteramos solo sobre las Clases de Equivalencia (Las tuplas únicas)
    for tupla, frecuencia in dict_estados.items():
        tupla_np = np.array(tupla, dtype=float)
        
        # Máscara para ignorar los NaNs en esta CE específica
        validos = ~np.isnan(tupla_np)
        
        # Si hay menos de 2 jueces válidos en esta tupla, no hay desacuerdo posible
        if np.sum(validos) < 2:
            continue
            
        # Extraemos solo los votos válidos
        votos_validos = tupla_np[validos]
        
        # 1. Calculamos el "centro" o consenso de esta Clase de Equivalencia
        if "Intervalar" in topologia:
            centro_ce = np.mean(votos_validos)
            # Desviación métrica (cuadrática o absoluta)
            desviaciones = (tupla_np - centro_ce) ** 2
        else:
            # Para Nominal y Ordinal, el centro es la moda (el valor más votado)
            # o calculamos el desacuerdo por pares (ej. cuántos no coinciden contigo)
            valores, conteos = np.unique(votos_validos, return_counts=True)
            moda = valores[np.argmax(conteos)]
            # Penalizamos si el juez no votó lo que votó la mayoría en esta CE
            desviaciones = np.where(tupla_np != moda, 1.0, 0.0)
            
        # 2. EL ATAJO MAESTRO: Multiplicamos la desviación de esta CE por su frecuencia real
        # Y acumulamos el resultado en el total de cada juez
        desviaciones_ponderadas = desviaciones * frecuencia
        
        # Sumamos con cuidado de no afectar a los NaNs
        np.add.at(penalizaciones_jueces, np.where(validos)[0], desviaciones_ponderadas[validos])
        np.add.at(total_votos_juez, np.where(validos)[0], frecuencia)

    # 3. Normalizamos la penalización por el número de veces que el juez realmente votó
    # para no penalizar/beneficiar artificialmente a los jueces con muchos datos faltantes
    indice_anomalia_final = np.divide(
        penalizaciones_jueces, 
        total_votos_juez, 
        out=np.zeros_like(penalizaciones_jueces), 
        where=total_votos_juez != 0
    )
    
    return indice_anomalia_final