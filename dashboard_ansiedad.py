import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador de Ansiedad", layout="centered")
st.title("🧠 Simulador de Ansiedad (TAG vs Ansiedad episódica)")

st.write("""
Este simulador educativo te permite visualizar cómo evoluciona la ansiedad según distintos factores personales, sociales y clínicos.  
**No reemplaza un diagnóstico profesional.**  
Puedes comparar un modelo matemático clásico y otro más realista (rebotes),  
y también personalizar tus propios umbrales y ver el rango de días para recuperación funcional.
""")

# ========== UMBRALES PERSONALIZADOS =========== 
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

# =================== AYUDA Y DIFERENCIAS DE MODELOS =====================
with st.expander("ℹ️ ¿Qué muestran los modelos clásico y realista?"):
    st.markdown("""
- **Modelo clásico:** Supone que la ansiedad siempre baja de forma continua, como una curva suave y predecible. Es útil para ver tendencias generales y el impacto de cada factor.
- **Modelo realista:** Agrega “rebotes” naturales en la ansiedad, reflejando cómo en la vida real hay altibajos (aunque haya mejoría general). Simula retrocesos, picos y el efecto del estrés sobre la recuperación.
- **¿Por qué comparar ambos?**  
  Porque la ansiedad humana rara vez es solo matemática: podemos mejorar pero igual tener días malos.  
  El modelo realista ayuda a “normalizar” esos rebotes y a no frustrarse si hay altibajos.
""")

# ======== 1️⃣ PERFIL PERSONAL (Sensibilidad) =========
st.header("1️⃣ Perfil Personal")
st.write("**Estos factores influyen en tu sensibilidad emocional actual.**")
trauma = st.slider(
    "Eventos traumáticos recientes",
    0, 5, 2,
    help="¿Has vivido algo difícil últimamente? (0 = nada; 5 = trauma intenso reciente)"
)
resiliencia = st.slider(
    "Resiliencia percibida",
    0, 5, 3,
    help="¿Qué tan rápido sientes que puedes reponerte de los golpes emocionales?"
)
regulacion = st.slider(
    "Capacidad para regular emociones",
    0, 5, 3,
    help="¿Te cuesta o no gestionar lo que sientes? (0 = muy difícil; 5 = lo gestionas bien)"
)
Sp = 1 + (trauma * 0.1) - (resiliencia * 0.05) - (regulacion * 0.05)
Sp = max(0.5, min(2.0, Sp))
st.markdown(f"🔎 **Sensibilidad personal calculada (Sp):** `{Sp:.2f}`")

# ========== 2️⃣ FACTORES INICIALES ===========
st.header("2️⃣ Factores de Ansiedad")
st.write("**Ajusta tu situación actual en la escala.**")
A0 = st.slider(
    "Nivel inicial de ansiedad",
    0, 100, 70,
    help="¿Cómo está tu ansiedad HOY? (0 = muy bajo, 100 = crisis total)"
)
E = st.slider(
    "Eventos estresantes",
    0, 100, 60,
    help="¿Cuánto estrés hay en tu día a día? (0 = nada; 100 = estrés brutal)"
)
S = st.slider(
    "Apoyo social",
    0, 100, 50,
    help="¿Qué tan acompañado(a) te sientes? (0 = solo/a; 100 = rodeado/a y sostenido/a)"
)
T_base = st.slider(
    "Terapia psicológica",
    0, 100, 40,
    help="¿Recibes terapia? (0 = nunca; 100 = tratamiento intensivo)"
)

st.write("🔹 Entre más alto tu apoyo social y terapia, más rápida suele ser la recuperación.")

# ========== 3️⃣ TRATAMIENTO MÉDICO ===========
toma_medicina = st.radio(
    "¿Tomas medicación para la ansiedad?",
    ("No", "Sí"),
    help="Incluye psicofármacos recetados. El modelo solo considera la variable global de medicación, no tipos específicos."
)
if toma_medicina == "Sí":
    M_base = st.slider(
        "Efecto de la medicación",
        0, 100, 30,
        help="Ajusta según dosis/intensidad (consulta siempre con profesional de salud)."
    )
else:
    M_base = 0

# ========== 4️⃣ COEFICIENTES Y MODELO ===========
expander = st.expander("⚙️ Opciones avanzadas: Ajusta los coeficientes (solo para expertos)")
with expander:
    k1 = st.slider("k1 (reducción natural)", 0.01, 0.15, 0.05, 0.01)
    k2 = st.slider("k2 (impacto eventos estresantes)", 0.01, 0.10, 0.04, 0.01)
    k3 = st.slider("k3 (efecto apoyo social)", 0.01, 0.07, 0.03, 0.01)
    k4 = st.slider("k4 (efecto terapia)", 0.01, 0.07, 0.02, 0.01)
    k5 = st.slider("k5 (efecto medicación)", 0.01, 0.15, 0.05, 0.01)
if not expander.expanded:
    k1, k2, k3, k4, k5 = 0.05, 0.04, 0.03, 0.02, 0.05

dias = 90

# ========== 5️⃣ FUNCIONES DE SIMULACIÓN ===========
def saturacion(x, max_ef=100, escala=30):
    return max_ef * (1 - np.exp(-x/escala))

def simular_realista(A0, E, S, T, M, Sp, dias, rebotes=True):
    A = np.zeros(dias)
    A[0] = A0
    b = 0.07 + (0.15 - Sp*0.04)
    w = 0.48 + (E/500)
    for t in range(1, dias):
        S_eff = saturacion(S, max_ef=100, escala=40)
        M_eff = saturacion(M, max_ef=100, escala=30)
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

# Modelos clásico/realista para comparativo:
escenario1 = simular_realista(A0, E, S, 0, 0, Sp, dias, rebotes=False)
escenario2 = simular_realista(A0, E, S, T_base, 0, Sp, dias, rebotes=False)
escenario3 = simular_realista(A0, E, S, T_base, M_base, Sp, dias, rebotes=False)
escenario1_osc = simular_realista(A0, E, S, 0, 0, Sp, dias, rebotes=True)
escenario2_osc = simular_realista(A0, E, S, T_base, 0, Sp, dias, rebotes=True)
escenario3_osc = simular_realista(A0, E, S, T_base, M_base, Sp, dias, rebotes=True)

def clasificar(A):
    promedio = np.mean(A)
    estado = "TAG" if promedio >= 60 else "Ansiedad episódica"
    return promedio, estado

prom1, est1 = clasificar(escenario1)
prom2, est2 = clasificar(escenario2)
prom3, est3 = clasificar(escenario3)
prom1o, est1o = clasificar(escenario1_osc)
prom2o, est2o = clasificar(escenario2_osc)
prom3o, est3o = clasificar(escenario3_osc)

# ========== 6️⃣ DIAS HASTA UMBRAL (escenario principal realista+rebotes) ===========
def dia_umbral(arr, umbral):
    for i, v in enumerate(arr):
        if v <= umbral:
            return i
    return None

escenario = simular_realista(A0, E, S, T_base, M_base, Sp, dias, rebotes=True)
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

# ========== 7️⃣ RESULTADOS Y CLASIFICACIÓN COMPARATIVA ==========
st.header("Resultados comparativos de modelos")
st.write("🔬 *Resultados promedio de los primeros 60 días para visualizar diferencias.*")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sin tratamiento", f"{prom1:.2f}", est1)
with col2:
    st.metric("Con terapia", f"{prom2:.2f}", est2)
with col3:
    st.metric("Terapia + medicación", f"{prom3:.2f}", est3)
st.write("**Umbral clínico estimado para TAG:** 60")

# ========== 8️⃣ GRÁFICOS EVOLUTIVOS ===========

st.header("Evolución comparativa de la Ansiedad (90 días)")

modelo = st.radio(
    "¿Qué modelo deseas visualizar?",
    [
        "Clásico (descenso suave y continuo)",
        "Realista (con rebotes y retrocesos)",
        "Solo tu escenario personalizado + umbrales"
    ],
    help="Compara modelos o solo revisa tu caso realista con zonas personalizadas."
)

fig, ax = plt.subplots()
if modelo == "Clásico (descenso suave y continuo)":
    ax.plot(range(dias), escenario1, label=f"Sin tratamiento ({est1})", linewidth=2)
    ax.plot(range(dias), escenario2, label=f"Con terapia ({est2})", linestyle='dashed', linewidth=2)
    ax.plot(range(dias), escenario3, label=f"Terapia + medicación ({est3})", linestyle='dotted', linewidth=2)
    ax.axhline(y=60, color='red', linestyle='--', label="Umbral TAG")
elif modelo == "Realista (con rebotes y retrocesos)":
    ax.plot(range(dias), escenario1_osc, label=f"Sin tratamiento ({est1o})", linewidth=2)
    ax.plot(range(dias), escenario2_osc, label=f"Con terapia ({est2o})", linestyle='dashed', linewidth=2)
    ax.plot(range(dias), escenario3_osc, label=f"Terapia + medicación ({est3o})", linestyle='dotted', linewidth=2)
    ax.axhline(y=60, color='red', linestyle='--', label="Umbral TAG")
else:
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
st.info("""
**Este simulador es educativo y no reemplaza atención profesional.  
Si tu ansiedad persiste o se agrava, busca acompañamiento con un especialista en salud mental.  
No te juzgues por los retrocesos: la recuperación es un camino con subidas y bajadas.**
""")

# ========== 1️⃣0️⃣ FÓRMULA USADA ===========
with st.expander("Ver detalles de la fórmula matemática"):
    st.latex(r"""
    \frac{dA}{dt} = -k_1 A(t) + Sp \cdot k_2 E - Sp(k_3 S + k_4 T + k_5 M)
    """)
    st.write("""
    Donde:
    - $A(t)$: nivel de ansiedad en el tiempo
    - $Sp$: sensibilidad personal (ajusta el impacto de factores)
    - $E$: eventos estresantes
    - $S$: apoyo social
    - $T$: terapia psicológica
    - $M$: medicación
    """)
