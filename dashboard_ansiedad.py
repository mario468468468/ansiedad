import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ============ CONFIGURACIÓN DE LA APP ==============
st.set_page_config(page_title="Simulador de Ansiedad", layout="centered")
st.title("🧠 Simulador de Ansiedad (TAG vs Ansiedad episódica)")
st.write(
    "Ajusta los valores y mira cómo evoluciona tu ansiedad en distintos escenarios: "
    "sin tratamiento, con terapia y con medicación. El modelo NO reemplaza diagnóstico clínico."
)

# ======== 1️⃣ PERFIL PERSONAL (Sensibilidad) =========
st.header("1️⃣ Perfil Personal")
st.write("Responde del 0 al 5 (0=nada, 5=muy alto):")
trauma = st.slider("Eventos traumáticos recientes", 0, 5, 2)
resiliencia = st.slider("Resiliencia percibida", 0, 5, 3)
regulacion = st.slider("Capacidad para regular emociones", 0, 5, 3)

Sp = 1 + (trauma * 0.1) - (resiliencia * 0.05) - (regulacion * 0.05)
Sp = max(0.5, min(2.0, Sp))
st.markdown(f"**Sensibilidad personal (Sp):** {Sp:.2f}")

# ========== 2️⃣ FACTORES INICIALES ===========
st.header("2️⃣ Factores iniciales")
st.write("0 = nada / sin efecto | 100 = máximo posible")
A0 = st.slider("Nivel inicial de ansiedad", 0, 100, 70)
E = st.slider("Eventos estresantes", 0, 100, 60)
S = st.slider("Apoyo social", 0, 100, 50)
T_base = st.slider("Terapia psicológica", 0, 100, 40)

# ========== 3️⃣ PREGUNTA SOBRE MEDICACIÓN ===========
st.header("3️⃣ Medición (Tratamiento médico)")
toma_medicina = st.radio("¿Tomas medicación?", ("No", "Sí"))
if toma_medicina == "Sí":
    M_base = st.slider("Efecto de la medicación", 0, 100, 30)
else:
    M_base = 0

# ========== 4️⃣ COEFICIENTES Y MODELO ===========
k1, k2, k3, k4, k5 = 0.05, 0.04, 0.03, 0.02, 0.05
A_TAG = 60
dias = 60

# ----------- MODELO AVANZADO: agrega oscilaciones amortiguadas -----------

def simular_realista(A0, E, S, T, M, Sp, dias, rebotes=True):
    """
    Simula ansiedad con decaimiento y, si rebotes=True, oscilaciones amortiguadas (modelo realista).
    """
    A = np.zeros(dias)
    A[0] = A0
    # Parámetros para la onda amortiguada
    b = 0.07 + (0.15 - Sp*0.04)         # amortiguamiento
    w = 0.48 + (E/500)                  # frecuencia (sube si hay mucho estrés)
    ruido_amp = 1.5 if rebotes else 0   # amplitud de “ruido emocional”
    for t in range(1, dias):
        # Término base (como tu modelo)
        dA_dt = -k1 * A[t-1] + Sp * k2 * E - Sp * (k3*S + k4*T + k5*M)
        base = A[t-1] + dA_dt
        if rebotes:
            # Onda amortiguada + “ruido emocional”
            osc = np.exp(-b*t) * np.cos(w*t) * (0.22*A0)
            ruido = np.random.normal(0, ruido_amp)
        else:
            osc = 0
            ruido = 0
        A[t] = max(0, base + osc + ruido)
    return A

# ------------ ESCENARIOS: clásico vs realista (con rebotes) -----------------
# Modelo clásico: sin rebotes
escenario1 = simular_realista(A0, E, S, 0, 0, Sp, dias, rebotes=False)
escenario2 = simular_realista(A0, E, S, T_base, 0, Sp, dias, rebotes=False)
escenario3 = simular_realista(A0, E, S, T_base, M_base, Sp, dias, rebotes=False)
# Modelo realista: con rebotes/ondas
escenario1_osc = simular_realista(A0, E, S, 0, 0, Sp, dias, rebotes=True)
escenario2_osc = simular_realista(A0, E, S, T_base, 0, Sp, dias, rebotes=True)
escenario3_osc = simular_realista(A0, E, S, T_base, M_base, Sp, dias, rebotes=True)

def clasificar(A):
    promedio = np.mean(A)
    estado = "TAG" if promedio >= A_TAG else "Ansiedad episódica"
    return promedio, estado

prom1, est1 = clasificar(escenario1)
prom2, est2 = clasificar(escenario2)
prom3, est3 = clasificar(escenario3)
prom1o, est1o = clasificar(escenario1_osc)
prom2o, est2o = clasificar(escenario2_osc)
prom3o, est3o = clasificar(escenario3_osc)

# ========= 6️⃣ RESULTADOS Y CLASIFICACIÓN ==========
st.header("4️⃣ Resultados y Diagnóstico Modelo")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sin tratamiento", f"{prom1:.2f}", est1)
with col2:
    st.metric("Con terapia", f"{prom2:.2f}", est2)
with col3:
    st.metric("Terapia + medicación", f"{prom3:.2f}", est3)
st.write("**Umbral clínico estimado para TAG:** 60")

# ========== 7️⃣ GRÁFICO EVOLUTIVO ===========

st.header("5️⃣ Evolución de la Ansiedad (60 días)")
modelo = st.radio("¿Qué modelo ver?", ["Clásico (descenso suave)", "Realista (con rebotes)"])
fig, ax = plt.subplots()
if modelo == "Clásico (descenso suave)":
    ax.plot(range(dias), escenario1, label=f"Sin tratamiento ({est1})", linewidth=2)
    ax.plot(range(dias), escenario2, label=f"Con terapia ({est2})", linestyle='dashed', linewidth=2)
    ax.plot(range(dias), escenario3, label=f"Terapia + medicación ({est3})", linestyle='dotted', linewidth=2)
else:
    ax.plot(range(dias), escenario1_osc, label=f"Sin tratamiento ({est1o})", linewidth=2)
    ax.plot(range(dias), escenario2_osc, label=f"Con terapia ({est2o})", linestyle='dashed', linewidth=2)
    ax.plot(range(dias), escenario3_osc, label=f"Terapia + medicación ({est3o})", linestyle='dotted', linewidth=2)
ax.axhline(y=A_TAG, color='red', linestyle='--', label="Umbral TAG")
ax.set_xlabel("Días")
ax.set_ylabel("Nivel de ansiedad")
ax.set_title("Simulación comparativa de Ansiedad")
ax.legend()
st.pyplot(fig)

# ========== 8️⃣ NOTA FINAL ===========
st.info("El simulador es una herramienta educativa y NO reemplaza diagnóstico profesional. Consulta a un especialista en salud mental para diagnóstico y tratamiento individualizado.")

# ========== 9️⃣ FÓRMULA USADA ===========
# (Descomenta si quieres mostrar la fórmula en LaTeX)
# st.markdown("""
# ---
# **Modelo base usado:**
#
# $$
# \\frac{dA}{dt} = -k_1 A(t) + Sp \\cdot k_2 E - Sp(k_3 S + k_4 T + k_5 M)
# $$
#
# Donde:
# - $A(t)$: nivel de ansiedad en el tiempo
# - $Sp$: sensibilidad personal (ajusta el impacto de factores)
# - $E$: eventos estresantes
# - $S$: apoyo social
# - $T$: terapia psicológica
# - $M$: medicación
# """)

