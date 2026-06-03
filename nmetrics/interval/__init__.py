"""
Módulo Intervalar de n-metrics.
Contiene los núcleos termodinámicos (Marco N) y estimadores clásicos
para variables continuas/intervalares.
"""

# 1. Importaciones del motor Termodinámico (Marco N)
from .ni_core import (
    detectar_anomalias_ni,
    calcular_azar_termodinamico_ni,
    calcular_estadisticas_ni_unificada,
    calcular_percentil_universal_ni
)

# 2. Importaciones del motor Alpha de Krippendorff Intervalar (AKI)
from .aki_core import (
    calcular_estadisticas_aki,
    calcular_aki_poblacion_asintotica
)

# 3. Importaciones del motor ICC(2,1)
from .icc21_core import (
    calcular_estadisticas_icc21,
    calcular_icc_poblacion_asintotica
)

# Definimos explícitamente qué funciones son accesibles públicamente
__all__ = [
    "detectar_anomalias_ni",
    "calcular_azar_termodinamico_ni",
    "calcular_estadisticas_ni_unificada",
    "calcular_percentil_universal_ni",
    "calcular_estadisticas_aki",
    "calcular_aki_poblacion_asintotica",
    "calcular_estadisticas_icc21",
    "calcular_icc_poblacion_asintotica"
]