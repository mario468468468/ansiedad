import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador de Ansiedad", layout="centered")
st.title("🧠 Simulador de Ansiedad (TAG vs Ansiedad episódica)")

st.write("""
Este simulador educativo te permite visualizar cómo evoluciona la ansiedad según distintos factores personales, sociales y clínicos.  
**No reemplaza un diagnóstico profesional.**  
Puedes comparar un modelo matemático clásico y otro más realista, que incluye rebotes emocionales típicos de la experiencia humana.
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

# ========== 3️⃣ PREGUNTA SOBRE MEDICACIÓN ===========
st.header("3️⃣ Tratamiento médico")
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
k1, k2, k3, k4, k5 = 0.05, 0.04, 0.03, 0.02, 0.05
A_TAG = 60
dias = 60

def simular_realista(A0, E, S, T, M, Sp, dias, rebotes=True):
    """
    Simula ansiedad con decaimiento y, si rebotes=True, oscilaciones amortiguadas (modelo realista).
    """
    A = np.zeros(dias)
    A[0] = A0
    b = 0.07 + (0.15 - Sp*0.04)         # amortiguamiento
    w = 0.48 + (E/500)                  # frecuencia
    ruido_amp = 1.5 if rebotes else 0
    for t in range(1, dias):
        dA_dt = -k1 * A[t-1] + Sp * k2 * E - Sp * (k3*S + k4*T + k5*M)
        base = A[t-1] + dA_dt
        if rebotes:
            osc = np.exp(-b*t) * np.cos(w*t) * (0.22*A0)
            ruido = np.random.normal(0, ruido_amp)
        else:
            osc = 0
            ruido = 0
        A[t] = max(0, base + osc + ruido)
    return A

# ESCENARIOS: clásico vs realista (con rebotes)
escenario1 = simular_realista(A0, E, S, 0, 0, Sp, dias, rebotes=False)
escenario2 = simular_realista(A0, E, S, T_base, 0, Sp, dias, rebotes=False)
escenario3 = simular_realista(A0, E, S, T_base, M_base, Sp, dias, rebotes=False)
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
st.write(
    "🔬 *Estos resultados son promedio de los 60 días. Recuerda: si tu ansiedad sube y baja, **NO es un fracaso**: la oscilación es parte del proceso humano.*"
)
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

modelo = st.radio(
    "¿Qué modelo deseas visualizar?",
    [
        "Clásico (descenso suave y continuo)",
        "Realista (con rebotes y retrocesos)"
    ],
    help="Puedes comparar la evolución matemática pura con una simulación más cercana a la experiencia real."
)
with st.expander("¿Por qué la ansiedad puede 'subir' incluso cuando estás mejorando?"):
    st.write("""
    La ansiedad real nunca baja perfecto. Hay días buenos y otros malos, picos inesperados o retrocesos temporales.  
    **Eso es normal.** Lo importante es la tendencia general y aprender a acompañarse en esos altibajos.
    """)

fig, ax = plt.subplots()
if modelo == "Clásico (descenso suave y continuo)":
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
st.info("""
**Este simulador es educativo y no reemplaza atención profesional.  
Si tu ansiedad persiste o se agrava, busca acompañamiento con un especialista en salud mental.  
No te juzgues por los retrocesos: la recuperación es un camino con subidas y bajadas.**
""")

# ========== 9️⃣ FÓRMULA USADA ===========
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

