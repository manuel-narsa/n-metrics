"""
Módulo Ordinal de n-metrics.
Contiene los núcleos termodinámicos (Marco N) y estimadores clásicos
para variables categóricas ordenadas (escala ordinal).
"""

# 1. Importaciones del motor Termodinámico Ordinal (NO)
from .no_core import (
    calcular_estadisticas_no_unificada,
    detectar_anomalias_no,
    analizar_termodinamica_no
)

# 2. Importaciones del motor Alpha de Krippendorff Ordinal (AKO)
from .ako_core import (
    calcular_estadisticas_ako,
    calcular_ako_poblacion_asintotica
)

# 3. Importaciones del motor W de Kendall
from .w_core import (
    calcular_estadisticas_w,
    calcular_w_poblacion_asintotica
)

__all__ = [
    "calcular_estadisticas_no_unificada",
    "detectar_anomalias_no",
    "analizar_termodinamica_no",
    "calcular_estadisticas_ako",
    "calcular_ako_poblacion_asintotica",
    "calcular_estadisticas_w",
    "calcular_w_poblacion_asintotica"
]