import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador de Ansiedad", layout="centered")
st.title("🧠 Simulador de Ansiedad (TAG vs Ansiedad episódica)")

st.write("""
Ajusta los valores para ver cómo evoluciona la ansiedad según diferentes factores.  
Puedes personalizar tus propios umbrales y ver un rango de días estimados para la recuperación.
""")

# ========== 1️⃣ UMBRALES PERSONALIZADOS ===========
st.header("Umbral Personalizado y Zonas")
A_OBJ = st.slider(
    "Define tu nivel de ansiedad funcional (objetivo)",
    0, 100, 40,
    help="¿Con qué nivel de ansiedad sientes que ya puedes vivir y funcionar bien?"
)
A_BAJA = st.slider("Umbral zona BAJA (ansiedad controlada)", 0, 100, 40)
A_MEDIA = st.slider("Umbral zona MEDIA (requiere atención)", 0, 100, 60)
A_ALTA = st.slider("Umbral zona ALTA (nivel crítico)", 0, 100, 80)

st.write(f"""
- **Zona BAJA:** Ansiedad ≤ {A_BAJA}
- **Zona MEDIA:** Ansiedad entre {A_BAJA+1} y {A_MEDIA}
- **Zona ALTA:** Ansiedad ≥ {A_MEDIA+1}
""")

# ========== 2️⃣ PERFIL PERSONAL (Sensibilidad) ===========
st.header("Perfil Personal")
trauma = st.slider("Eventos traumáticos recientes", 0, 5, 2)
resiliencia = st.slider("Resiliencia percibida", 0, 5, 3)
regulacion = st.slider("Capacidad para regular emociones", 0, 5, 3)
Sp = 1 + (trauma * 0.1) - (resiliencia * 0.05) - (regulacion * 0.05)
Sp = max(0.5, min(2.0, Sp))
st.markdown(f"**Sensibilidad personal (Sp):** `{Sp:.2f}`")

# ========== 3️⃣ FACTORES INICIALES ===========
st.header("Factores de Ansiedad")
A0 = st.slider("Nivel inicial de ansiedad", 0, 100, 70)
E = st.slider("Eventos estresantes", 0, 100, 60)
S = st.slider("Apoyo social", 0, 100, 50)
T_base = st.slider("Terapia psicológica", 0, 100, 40)

st.write("🔹 Entre más alto tu apoyo social y terapia, más rápida suele ser la recuperación.")

# ========== 4️⃣ TRATAMIENTO MÉDICO ===========
toma_medicina = st.radio("¿Tomas medicación para la ansiedad?", ("No", "Sí"))
if toma_medicina == "Sí":
    M_base = st.slider("Efecto de la medicación", 0, 100, 30)
else:
    M_base = 0

# ========== 5️⃣ PONDERACIONES CONFIGURABLES (AVANZADO) ===========
with st.expander("⚙️ Opciones avanzadas: Ajusta los coeficientes (solo para expertos)"):
    k1 = st.slider("k1 (reducción natural)", 0.01, 0.15, 0.05, 0.01)
    k2 = st.slider("k2 (impacto eventos estresantes)", 0.01, 0.10, 0.04, 0.01)
    k3 = st.slider("k3 (efecto apoyo social)", 0.01, 0.07, 0.03, 0.01)
    k4 = st.slider("k4 (efecto terapia)", 0.01, 0.07, 0.02, 0.01)
    k5 = st.slider("k5 (efecto medicación)", 0.01, 0.15, 0.05, 0.01)

if 'k1' not in locals():
    k1, k2, k3, k4, k5 = 0.05, 0.04, 0.03, 0.02, 0.05

dias = 90

# ========== 6️⃣ SIMULADOR MEJORADO ===========
def saturacion(x, max_ef=100, escala=30):
    # Efecto no lineal de apoyo y medicación (saturación tipo log)
    return max_ef * (1 - np.exp(-x/escala))

def simular_realista(A0, E, S, T, M, Sp, dias, rebotes=True):
    A = np.zeros(dias)
    A[0] = A0
    b = 0.07 + (0.15 - Sp*0.04)
    w = 0.48 + (E/500)
    for t in range(1, dias):
        # Efecto no lineal en medicación y apoyo social
        S_eff = saturacion(S, max_ef=100, escala=40)
        M_eff = saturacion(M, max_ef=100, escala=30)
        # Feedback emocional: si ansiedad > 70 muchos días, Sp sube
        if t > 10 and np.mean(A[max(0, t-10):t]) > 70:
            Sp_t = Sp + 0.15
        else:
            Sp_t = Sp
        dA_dt = -k1 * A[t-1] + Sp_t * k2 * E - Sp_t * (k3*S_eff + k4*T + k5*M_eff)
        base = A[t-1] + dA_dt
        if rebotes:
            osc = np.exp(-b*t) * np.cos(w*t) * (0.18*A0)
            ruido = np.random.normal(0, 1.3)
        else:
            osc = 0
            ruido = 0
        A[t] = max(0, base + osc + ruido)
    return A

escenario = simular_realista(A0, E, S, T_base, M_base, Sp, dias, rebotes=True)

# ========== 7️⃣ DÍAS HASTA UMBRAL ===========
def dia_umbral(arr, umbral):
    for i, v in enumerate(arr):
        if v <= umbral:
            return i
    return None

dia_baja = dia_umbral(escenario, A_BAJA)
dia_media = dia_umbral(escenario, A_MEDIA)
dia_obj = dia_umbral(escenario, A_OBJ)

st.subheader("⏳ Tiempo estimado de recuperación (orientativo)")
if dia_obj:
    variab = int(5 + np.abs(Sp-1)*6 + np.abs(E-50)/20)
    rango_min = max(0, dia_obj - variab)
    rango_max = min(dias-1, dia_obj + variab)
    st.write(f"Llegar a tu objetivo ({A_OBJ}): entre **{rango_min} y {rango_max} días**.")
else:
    st.write("No se estima cruce de umbral objetivo en los 90 días simulados.")

if dia_baja:
    st.write(f"Zona BAJA: se alcanza en el día ~{dia_baja}.")
if dia_media:
    st.write(f"Zona MEDIA: se alcanza en el día ~{dia_media}.")

# ========== 8️⃣ GRÁFICO EVOLUTIVO Y UMBRALES ===========
fig, ax = plt.subplots()
ax.plot(range(dias), escenario, label="Ansiedad simulada", linewidth=2)
ax.axhline(y=A_BAJA, color='green', linestyle='--', label="Umbral BAJO")
ax.axhline(y=A_MEDIA, color='orange', linestyle='--', label="Umbral MEDIO")
ax.axhline(y=A_ALTA, color='red', linestyle='--', label="Umbral ALTO")
ax.axhline(y=A_OBJ, color='blue', linestyle=':', label="Tu objetivo")
ax.set_xlabel("Días")
ax.set_ylabel("Nivel de ansiedad")
ax.set_title("Evolución simulada de Ansiedad y zonas")
ax.legend()
st.pyplot(fig)

# ========== 9️⃣ NOTA FINAL ===========
st.info("Este simulador es educativo. La ansiedad tiene matices individuales y no reemplaza diagnóstico profesional.")


