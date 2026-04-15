import pandas as pd
import numpy as np
from scipy.signal import find_peaks 


def gait_events_simple(df):
    """
    Detecta Heel Strikes (HS) y Toe‐Offs (TO) basándose únicamente en las transiciones
    de los canales de contacto ("Contact RT" y "Contact LT").

    Asume que en df["Contact RT"] y df["Contact LT"]:
      - 0  = pie en el aire
      - 1000 = pie en apoyo

    Retorna cuatro listas de índices (enteros):
      hs_R: índices donde ocurre Heel‐Strike derecho
      hs_L: índices donde ocurre Heel‐Strike izquierdo
      to_R: índices donde ocurre Toe‐Off derecho
      to_L: índices donde ocurre Toe‐Off izquierdo
    """
    
    contact_R = df["Contact RT"].values > 0
    contact_L = df["Contact LT"].values > 0

    heel_strike_R = np.where((~contact_R[:-1]) & (contact_R[1:]))[0] + 1
    heel_strike_L = np.where((~contact_L[:-1]) & (contact_L[1:]))[0] + 1

    toe_off_R = np.where((contact_R[:-1]) & (~contact_R[1:]))[0] + 1
    toe_off_L = np.where((contact_L[:-1]) & (~contact_L[1:]))[0] + 1

    return heel_strike_R.tolist(), heel_strike_L.tolist(), toe_off_R.tolist(), toe_off_L.tolist()