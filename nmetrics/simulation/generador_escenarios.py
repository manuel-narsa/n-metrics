import numpy as np

def generar_matriz_dinamica(n_sujetos, m_evaluadores, k_escala, escenario, tipo_escala='categórica',
                            decimales=1, # <--- NUEVO: Controla la generación de valores continuos
                            # Parámetros ajustables NI
                            NICN=0.05, NIRT=0.50, NIRV=0.50, NICPT=0.30, NICPV=0.20, NICIT=0.10, NICIV=0.10,
                            # Parámetros ajustables NN
                            NNCN=0.05, NNRT=0.50, NNRV=0.50, NNCPT=0.30, NNCPV=0.20, NNCIT=0.10, NNCIV=0.10,
                            # Parámetros ajustables NO
                            NOCN=0.05, NORT=0.60, NORV=0.60, NOCPT=0.30, NOCPV=0.30, NOCIT=0.10, NOCIV=0.10):
    
    # Inicializamos la matriz base
    matriz = np.zeros((n_sujetos, m_evaluadores), dtype=int)

    # =========================================================================
    # LÓGICA COMPARTIDA PARA: Razonable, Casi Perfecto, Casi Idéntico
    # =========================================================================
    def aplicar_consenso_y_ruido(pct_tuplas, pct_valores):
        base = np.random.randint(1, k_escala + 1, size=(n_sujetos, 1))
        mat = np.tile(base, (1, m_evaluadores))
        
        num_tuplas = max(1, int(pct_tuplas * n_sujetos)) if pct_tuplas > 0 else 0
        num_valores = max(1, int(pct_valores * m_evaluadores)) if pct_valores > 0 else 0
        
        if num_tuplas > 0 and num_valores > 0:
            tuplas_a_modificar = np.random.choice(n_sujetos, num_tuplas, replace=False)
            for t in tuplas_a_modificar:
                cols_a_modificar = np.random.choice(m_evaluadores, num_valores, replace=False)
                for c in cols_a_modificar:
                    val_actual = mat[t, c]
                    opciones = [x for x in range(1, k_escala + 1) if x != val_actual]
                    mat[t, c] = np.random.choice(opciones)
        return mat

    # =========================================================================
    # GENERACIÓN BASE SEGÚN ESCENARIO Y TIPO
    # =========================================================================
    if escenario == "Aleatorio":
        matriz = np.random.randint(1, k_escala + 1, size=(n_sujetos, m_evaluadores))

    elif tipo_escala == 'paramétrica':
        if escenario == "Casi Nulo":
            mitad = m_evaluadores // 2
            for i in range(n_sujetos):
                fila = [1] * mitad + [k_escala] * (m_evaluadores - mitad)
                np.random.shuffle(fila)
                matriz[i, :] = fila
            
            num_tuplas = max(1, int(NICN * n_sujetos)) if NICN > 0 else 0
            if k_escala > 2 and num_tuplas > 0: 
                tuplas_a_modificar = np.random.choice(n_sujetos, num_tuplas, replace=False)
                for t in tuplas_a_modificar:
                    col = np.random.randint(0, m_evaluadores)
                    matriz[t, col] = np.random.randint(2, k_escala)
        elif escenario == "Razonable":
            matriz = aplicar_consenso_y_ruido(NIRT, NIRV)
        elif escenario == "Casi Perfecto":
            matriz = aplicar_consenso_y_ruido(NICPT, NICPV)
        elif escenario == "Casi Idéntico":
            matriz = aplicar_consenso_y_ruido(NICIT, NICIV)

    elif tipo_escala == 'categórica':
        if escenario == "Casi Nulo":
            secuencia = np.resize(np.arange(1, k_escala + 1), n_sujetos * m_evaluadores)
            matriz = secuencia.reshape(n_sujetos, m_evaluadores)
            num_tuplas = max(1, int(NNCN * n_sujetos)) if NNCN > 0 else 0
            if num_tuplas > 0:
                tuplas_a_modificar = np.random.choice(n_sujetos, num_tuplas, replace=False)
                for t in tuplas_a_modificar:
                    col = np.random.randint(0, m_evaluadores)
                    val_actual = matriz[t, col]
                    opciones = [x for x in range(1, k_escala + 1) if x != val_actual]
                    matriz[t, col] = np.random.choice(opciones)
        elif escenario == "Razonable":
            matriz = aplicar_consenso_y_ruido(NNRT, NNRV)
        elif escenario == "Casi Perfecto":
            matriz = aplicar_consenso_y_ruido(NNCPT, NNCPV)
        elif escenario == "Casi Idéntico":
            matriz = aplicar_consenso_y_ruido(NNCIT, NNCIV)

    elif tipo_escala == 'ordinal':
        if escenario == "Casi Nulo":
            secuencia = np.resize(np.arange(1, k_escala + 1), n_sujetos * m_evaluadores)
            matriz = secuencia.reshape(n_sujetos, m_evaluadores)
            num_tuplas = max(1, int(NOCN * n_sujetos)) if NOCN > 0 else 0
            if num_tuplas > 0:
                tuplas_a_modificar = np.random.choice(n_sujetos, num_tuplas, replace=False)
                for t in tuplas_a_modificar:
                    col = np.random.randint(0, m_evaluadores)
                    val_actual = matriz[t, col]
                    opciones = [x for x in range(1, k_escala + 1) if x != val_actual]
                    matriz[t, col] = np.random.choice(opciones)
        elif escenario == "Razonable":
            matriz = aplicar_consenso_y_ruido(NORT, NORV)
        elif escenario == "Casi Perfecto":
            matriz = aplicar_consenso_y_ruido(NOCPT, NOCPV)
        elif escenario == "Casi Idéntico":
            matriz = aplicar_consenso_y_ruido(NOCIT, NOCIV)

    else:
        raise ValueError(f"Tipo de escala '{tipo_escala}' no reconocido.")

    # =========================================================================
    # APLICACIÓN DE DECIMALES (RUIDO CONTINUO)
    # =========================================================================
    if decimales > 0:
        matriz = matriz.astype(float)
        # Se añade un ruido uniforme entre -0.49 y +0.49 a las puntuaciones base
        ruido = np.random.uniform(-0.49, 0.49, size=matriz.shape)
        matriz += ruido
        # Aseguramos que los valores no excedan los límites de la escala [1, k_escala]
        matriz = np.clip(matriz, 1.0, float(k_escala))
        # Redondeamos a los decimales solicitados
        matriz = np.round(matriz, decimales)

    return matriz