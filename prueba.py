# filename: dashboard_ansiedad.py (colócalo en la raíz del proyecto)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ============================ CONFIG =============================
st.set_page_config(page_title="Simulador de Ansiedad", layout="centered")
st.title("🧠 Simulador de Ansiedad (TAG vs Ansiedad episódica)")

st.write(
    """
Este simulador educativo te permite visualizar cómo evoluciona la ansiedad según distintos factores personales, sociales y clínicos.  
**No reemplaza un diagnóstico profesional.**  
Puedes comparar un modelo matemático clásico, otro más realista (con rebotes), y **un modelo de cadenas de Markov (regímenes discretos)** que estima **probabilidades** de estar en cada zona (BAJA/MEDIA/ALTA) día a día.
"""
)

# ===================== 0) UTILIDADES GENERALES ====================
np.set_printoptions(precision=4, suppress=True)

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

# ========== UMBRALES PERSONALIZADOS =========== 
st.header("Umbral Personalizado y Zonas")
A_OBJ = st.slider(
    "Define tu nivel de ansiedad funcional (objetivo)",
    0, 100, 40,
    help="¿Con qué nivel de ansiedad sientes que ya puedes vivir y funcionar bien?",
)
A_BAJA = st.slider("Umbral zona BAJA (ansiedad controlada)", 0, 100, 40)
A_MEDIA = st.slider("Umbral zona MEDIA (requiere atención)", 0, 100, 60)
A_ALTA = st.slider("Umbral zona ALTA (nivel crítico)", 0, 100, 80)

st.write(
    f"""
- **Zona BAJA:** Ansiedad ≤ {A_BAJA}
- **Zona MEDIA:** Ansiedad entre {A_BAJA+1} y {A_MEDIA}
- **Zona ALTA:** Ansiedad ≥ {A_MEDIA+1}
"""
)

# =================== AYUDA Y DIFERENCIAS DE MODELOS =====================
with st.expander("ℹ️ ¿Qué muestran los modelos clásico, realista y Markov?"):
    st.markdown(
        """
- **Modelo clásico (ODE):** Supone que la ansiedad tiende a bajar de forma continua, como una curva suave y predecible (tendencia media).
- **Modelo realista (ODE + rebotes):** Agrega oscilaciones y ruido (altibajos naturales). Simula retrocesos, picos y el efecto del estrés.
- **Modelo de cadenas de Markov (regímenes):** Describe **probabilidades** de estar en **zonas discretas (BAJA/MEDIA/ALTA)** y cómo **cambian** con los factores (estrés, apoyo, terapia, medicación, sensibilidad). Permite estimar el **día probable** en que alcanzas tu objetivo.
- **¿Por qué añadir Markov?** Porque la ansiedad real cambia por **regímenes** (buenos/malos días). Markov captura **persistencia** e **inercia** de estados y cuantifica la **incertidumbre** día a día.
"""
    )

# ======== 1️⃣ PERFIL PERSONAL (Sensibilidad) =========
st.header("1️⃣ Perfil Personal")
st.write("**Estos factores influyen en tu sensibilidad emocional actual.**")
trauma = st.slider(
    "Eventos traumáticos recientes",
    0, 5, 2,
    help="¿Has vivido algo difícil últimamente? (0 = nada; 5 = trauma intenso reciente)",
)
resiliencia = st.slider(
    "Resiliencia percibida",
    0, 5, 3,
    help="¿Qué tan rápido sientes que puedes reponerte de los golpes emocionales?",
)
regulacion = st.slider(
    "Capacidad para regular emociones",
    0, 5, 3,
    help="¿Te cuesta o no gestionar lo que sientes? (0 = muy difícil; 5 = lo gestionas bien)",
)
Sp = 1 + (trauma * 0.1) - (resiliencia * 0.05) - (regulacion * 0.05)
Sp = max(0.5, min(2.0, Sp))
st.markdown(f"🔎 **Sensibilidad personal calculada (Sp):** `{Sp:.2f}`")

# ========== 2️⃣ FACTORES DE ANSIEDAD ===========
st.header("2️⃣ Factores de Ansiedad")
st.write("**Ajusta tu situación actual en la escala.**")
A0 = st.slider(
    "Nivel inicial de ansiedad",
    0, 100, 70,
    help="¿Cómo está tu ansiedad HOY? (0 = muy bajo, 100 = crisis total)",
)
E = st.slider(
    "Eventos estresantes",
    0, 100, 60,
    help="¿Cuánto estrés hay en tu día a día? (0 = nada; 100 = estrés brutal)",
)
S = st.slider(
    "Apoyo social",
    0, 100, 50,
    help="¿Qué tan acompañado(a) te sientes? (0 = solo/a; 100 = rodeado/a y sostenido/a)",
)
T_base = st.slider(
    "Terapia psicológica",
    0, 100, 40,
    help="¿Recibes terapia? (0 = nunca; 100 = tratamiento intensivo)",
)

st.write("🔹 Entre más alto tu apoyo social y terapia, más rápida suele ser la recuperación.")

# ========== 3️⃣ TRATAMIENTO MÉDICO ===========
toma_medicina = st.radio(
    "¿Tomas medicación para la ansiedad?",
    ("No", "Sí"),
    help="Incluye psicofármacos recetados. El modelo solo considera la variable global de medicación, no tipos específicos.",
)
if toma_medicina == "Sí":
    M_base = st.slider(
        "Efecto de la medicación",
        0, 100, 30,
        help="Ajusta según dosis/intensidad (consulta siempre con profesional de salud).",
    )
else:
    M_base = 0

# ========== 4️⃣ COEFICIENTES Y MODELO ODE ===========
expander = st.expander("⚙️ Opciones avanzadas ODE (solo para expertos)")
with expander:
    k1 = st.slider("k1 (reducción natural)", 0.01, 0.15, 0.05, 0.01)
    k2 = st.slider("k2 (impacto eventos estresantes)", 0.01, 0.10, 0.04, 0.01)
    k3 = st.slider("k3 (efecto apoyo social)", 0.01, 0.07, 0.03, 0.01)
    k4 = st.slider("k4 (efecto terapia)", 0.01, 0.07, 0.02, 0.01)
    k5 = st.slider("k5 (efecto medicación)", 0.01, 0.15, 0.05, 0.01)
if not expander.expanded:
    k1, k2, k3, k4, k5 = 0.05, 0.04, 0.03, 0.02, 0.05

# Horizonte temporal
dias = 90

# =================== 5️⃣ FUNCIONES ODE (clásico/realista) ===================
def saturacion(x, max_ef=100, escala=30):
    return max_ef * (1 - np.exp(-x / escala))


def simular_realista(A0, E, S, T, M, Sp, dias, rebotes=True):
    A = np.zeros(dias)
    A[0] = A0
    b = 0.07 + (0.15 - Sp * 0.04)  # amortiguación de oscilación
    w = 0.48 + (E / 500)  # frecuencia (ligada a estrés)
    for t in range(1, dias):
        S_eff = saturacion(S, max_ef=100, escala=40)
        M_eff = saturacion(M, max_ef=100, escala=30)

        # Efecto de sobrecarga acumulada: eleva temporalmente la sensibilidad
        if t > 10 and np.mean(A[max(0, t - 10) : t]) > 70:
            Sp_t = Sp + 0.15
        else:
            Sp_t = Sp

        dA_dt = -k1 * A[t - 1] + Sp_t * k2 * E - Sp_t * (k3 * S_eff + k4 * T + k5 * M_eff)
        base = A[t - 1] + dA_dt

        if rebotes:
            osc = np.exp(-b * t) * np.cos(w * t) * (0.18 * A0)
            ruido = np.random.normal(0, 1.3)
        else:
            osc = 0
            ruido = 0
        A[t] = max(0, base + osc + ruido)
    return A


# Modelos ODE (clásico/realista) para comparativo:
escenario1 = simular_realista(A0, E, S, 0, 0, Sp, dias, rebotes=False)
escenario2 = simular_realista(A0, E, S, T_base, 0, Sp, dias, rebotes=False)
escenario3 = simular_realista(A0, E, S, T_base, M_base, Sp, dias, rebotes=False)

escenario1_osc = simular_realista(A0, E, S, 0, 0, Sp, dias, rebotes=True)
escenario2_osc = simular_realista(A0, E, S, T_base, 0, Sp, dias, rebotes=True)
escenario3_osc = simular_realista(A0, E, S, T_base, M_base, Sp, dias, rebotes=True)


def clasificar(A):
    promedio = np.mean(A[:60])  # 60 días para comparar
    estado = "TAG" if promedio >= 60 else "Ansiedad episódica"
    return promedio, estado


prom1, est1 = clasificar(escenario1)
prom2, est2 = clasificar(escenario2)
prom3, est3 = clasificar(escenario3)
prom1o, est1o = clasificar(escenario1_osc)
prom2o, est2o = clasificar(escenario2_osc)
prom3o, est3o = clasificar(escenario3_osc)

# ========== 6️⃣ DÍAS HASTA UMBRAL (escenario principal ODE realista+rebotes) ==========

def dia_umbral(arr, umbral):
    for i, v in enumerate(arr):
        if v <= umbral:
            return i
    return None


escenario = simular_realista(A0, E, S, T_base, M_base, Sp, dias, rebotes=True)
dia_baja = dia_umbral(escenario, A_BAJA)
dia_media = dia_umbral(escenario, A_MEDIA)
dia_obj = dia_umbral(escenario, A_OBJ)

st.subheader("⏳ Tiempo estimado de recuperación (orientativo) — Modelo ODE realista")
if dia_obj is not None:
    variab = int(5 + abs(Sp - 1) * 6 + abs(E - 50) / 20)
    rango_min = max(0, dia_obj - variab)
    rango_max = min(dias - 1, dia_obj + variab)
    st.write(f"Llegar a tu objetivo ({A_OBJ}): entre **{rango_min} y {rango_max} días**.")
else:
    st.write("No se estima cruce de umbral objetivo en los 90 días simulados.")

if dia_baja is not None:
    st.write(f"Zona BAJA: se alcanza en el día ~{dia_baja}.")
if dia_media is not None:
    st.write(f"Zona MEDIA: se alcanza en el día ~{dia_media}.")

# ===================== 7️⃣ PARÁMETROS MARKOV =====================
st.header("🧩 Modelo de Cadenas de Markov (regímenes BAJA/MEDIA/ALTA)")
with st.expander("⚙️ Parámetros del modelo Markov"):
    st.markdown(
        "Ajusta los promedios de ansiedad asociados a cada **estado** y la **inercia** (tendencia a permanecer) y **pesos** de los factores."
    )
    mu_L = st.slider("μ_BAJA (valor esperado en BAJA)", 0, 60, 30)
    mu_M = st.slider("μ_MEDIA (valor esperado en MEDIA)", 30, 80, 55)
    mu_H = st.slider("μ_ALTA (valor esperado en ALTA)", 60, 100, 80)

    inertia = st.slider("Inercia (κ): tendencia a permanecer en el mismo estado", 0.0, 3.0, 1.0, 0.1)

    st.write("**Pesos de factores (afectan transición a BAJA y ALTA; MEDIA es referencia).**")
    wE = st.slider("wE (estrés → ALTA, ← BAJA)", 0.0, 3.0, 1.0, 0.1)
    wS = st.slider("wS (apoyo → BAJA, ← ALTA)", 0.0, 3.0, 1.0, 0.1)
    wT = st.slider("wT (terapia → BAJA, ← ALTA)", 0.0, 3.0, 0.8, 0.1)
    wM = st.slider("wM (medicación → BAJA, ← ALTA)", 0.0, 3.0, 1.2, 0.1)
    wSp = st.slider("wSp (sensibilidad Sp → ALTA, ← BAJA)", 0.0, 3.0, 1.0, 0.1)

    bL = st.slider("bL (sesgo base hacia BAJA)", -2.0, 2.0, 0.0, 0.1)
    bM = st.slider("bM (sesgo base hacia MEDIA)", -2.0, 2.0, 0.0, 0.1)
    bH = st.slider("bH (sesgo base hacia ALTA)", -2.0, 2.0, 0.0, 0.1)

params_markov = {
    "mu": np.array([mu_L, mu_M, mu_H], dtype=float),
    "inertia": inertia,
    "wE": wE,
    "wS": wS,
    "wT": wT,
    "wM": wM,
    "wSp": wSp,
    "bL": bL,
    "bM": bM,
    "bH": bH,
}


def init_dist_from_A0(A0: float, A_BAJA: float, A_MEDIA: float) -> np.ndarray:
    """Distribución inicial en función del A0 y los umbrales."""
    if A0 <= A_BAJA:
        return np.array([1.0, 0.0, 0.0])
    elif A0 <= A_MEDIA:
        return np.array([0.0, 1.0, 0.0])
    else:
        return np.array([0.0, 0.0, 1.0])


def markov_row(curr_idx: int, E: float, S: float, T: float, M: float, Sp: float, p: dict) -> np.ndarray:
    """Construye la fila de transición (prob. a BAJA/MEDIA/ALTA) desde el estado curr_idx.

    - Inercia κ favorece quedarse en el mismo estado.
    - Pesos (wE, wS, wT, wM, wSp) afectan **tendencia** hacia BAJA o ALTA.
    - MEDIA se usa como referencia (sesgo bM).
    """
    scale = 1.0 / 100.0
    xE, xS, xT, xM = E * scale, S * scale, T * scale, M * scale

    # Puntuaciones hacia cada estado destino (logits)
    sL = p["bL"] + (-p["wE"] * xE + p["wS"] * xS + p["wT"] * xT + p["wM"] * xM - p["wSp"] * (Sp - 1))
    sM = p["bM"]
    sH = p["bH"] + (+p["wE"] * xE - p["wS"] * xS - p["wT"] * xT - p["wM"] * xM + p["wSp"] * (Sp - 1))

    # Inercia (κ) aumenta la probabilidad de quedarse donde estás
    if curr_idx == 0:
        sL += p["inertia"]
    elif curr_idx == 1:
        sM += p["inertia"]
    else:
        sH += p["inertia"]

    logits = np.array([sL, sM, sH], dtype=float)
    return softmax(logits)


def simulate_markov(A0: float, E: float, S: float, T: float, M: float, Sp: float, dias: int, p: dict):
    """Simula la evolución de **probabilidades** de estados (BAJA/MEDIA/ALTA)
    y el valor esperado E[A_t] usando μ de cada estado.
    """
    mu = p["mu"]  # [μ_L, μ_M, μ_H]

    # Distribución inicial basada en A0
    dist = init_dist_from_A0(A0, A_BAJA, A_MEDIA)

    probs = np.zeros((dias, 3))
    expA = np.zeros(dias)

    for t in range(dias):
        probs[t] = dist
        expA[t] = np.dot(mu, dist)

        # Construir matriz de transición tiempo-t (depende de factores)
        row_L = markov_row(0, E, S, T, M, Sp, p)
        row_M = markov_row(1, E, S, T, M, Sp, p)
        row_H = markov_row(2, E, S, T, M, Sp, p)
        P_t = np.vstack([row_L, row_M, row_H])  # 3x3

        # Actualizar distribución
        dist = dist @ P_t

    return probs, expA


probs_markov, expA_markov = simulate_markov(A0, E, S, T_base, M_base, Sp, dias, params_markov)

# ====== 7.1) Estimación de día objetivo por probabilidad (Markov) ======
def day_prob_reaching_goal(probs: np.ndarray, mu: np.ndarray, A_OBJ: float, q: float = 0.5):
    """Devuelve el primer día t en el que la prob. de estar en estados con μ <= A_OBJ
    alcanza al menos q (por defecto mediana q=0.5). Si no se alcanza, retorna None.
    """
    goal_states = np.where(mu <= A_OBJ)[0]
    if len(goal_states) == 0:
        return None
    cum = np.sum(probs[:, goal_states], axis=1)
    for t, pgoal in enumerate(cum):
        if pgoal >= q:
            return t
    return None


mu_vec = params_markov["mu"]
day_q25 = day_prob_reaching_goal(probs_markov, mu_vec, A_OBJ, q=0.25)
day_q50 = day_prob_reaching_goal(probs_markov, mu_vec, A_OBJ, q=0.50)
day_q75 = day_prob_reaching_goal(probs_markov, mu_vec, A_OBJ, q=0.75)

st.subheader("🎯 Probabilidad de alcanzar el objetivo — Modelo Markov")
if day_q50 is not None:
    rango_txt = []
    if day_q25 is not None:
        rango_txt.append(f"P25≈{day_q25}")
    if day_q75 is not None:
        rango_txt.append(f"P75≈{day_q75}")
    rango_str = f" (rango {', '.join(rango_txt)})" if rango_txt else ""
    st.write(f"**Día mediano (50%)** para estar en estados ≤ objetivo: **{day_q50}**{rango_str}.")
else:
    st.write("No se alcanza el 50% de probabilidad de estar ≤ objetivo en 90 días.")

# ========== 8️⃣ RESULTADOS Y CLASIFICACIÓN COMPARATIVA (ODE) ==========
st.header("Resultados comparativos de modelos (ODE)")
st.write("🔬 *Resultados promedio de los primeros 60 días para visualizar diferencias.*")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sin tratamiento", f"{prom1:.2f}", est1)
with col2:
    st.metric("Con terapia", f"{prom2:.2f}", est2)
with col3:
    st.metric("Terapia + medicación", f"{prom3:.2f}", est3)
st.write("**Umbral clínico estimado para TAG:** 60")

# ========== 9️⃣ GRÁFICOS EVOLUTIVOS ===========
st.header("Evolución comparativa de la Ansiedad (90 días)")

modelo = st.radio(
    "¿Qué modelo deseas visualizar?",
    [
        "Clásico (descenso suave y continuo)",
        "Realista (con rebotes y retrocesos)",
        "Cadena de Markov (regímenes discretos)",
        "Solo tu escenario ODE + umbrales",
    ],
    help="Compara modelos o revisa tu caso con zonas personalizadas.",
)

fig, ax = plt.subplots()
if modelo == "Clásico (descenso suave y continuo)":
    ax.plot(range(dias), escenario1, label=f"Sin tratamiento ({est1})", linewidth=2)
    ax.plot(range(dias), escenario2, label=f"Con terapia ({est2})", linestyle="dashed", linewidth=2)
    ax.plot(range(dias), escenario3, label=f"Terapia + medicación ({est3})", linestyle="dotted", linewidth=2)
    ax.axhline(y=60, color="red", linestyle="--", label="Umbral TAG")
elif modelo == "Realista (con rebotes y retrocesos)":
    ax.plot(range(dias), escenario1_osc, label=f"Sin tratamiento ({est1o})", linewidth=2)
    ax.plot(range(dias), escenario2_osc, label=f"Con terapia ({est2o})", linestyle="dashed", linewidth=2)
    ax.plot(range(dias), escenario3_osc, label=f"Terapia + medicación ({est3o})", linestyle="dotted", linewidth=2)
    ax.axhline(y=60, color="red", linestyle="--", label="Umbral TAG")
elif modelo == "Cadena de Markov (regímenes discretos)":
    ax.plot(range(dias), expA_markov, label="E[A_t] según Markov", linewidth=2)
    ax.axhline(y=A_BAJA, color="green", linestyle="--", label="Umbral BAJO")
    ax.axhline(y=A_MEDIA, color="orange", linestyle="--", label="Umbral MEDIO")
    ax.axhline(y=A_ALTA, color="red", linestyle="--", label="Umbral ALTO")
    ax.axhline(y=A_OBJ, color="blue", linestyle=":", label="Tu objetivo")
else:  # Solo escenario ODE + umbrales
    ax.plot(range(dias), escenario, label="Ansiedad simulada (ODE)", linewidth=2)
    ax.axhline(y=A_BAJA, color="green", linestyle="--", label="Umbral BAJO")
    ax.axhline(y=A_MEDIA, color="orange", linestyle="--", label="Umbral MEDIO")
    ax.axhline(y=A_ALTA, color="red", linestyle="--", label="Umbral ALTO")
    ax.axhline(y=A_OBJ, color="blue", linestyle=":", label="Tu objetivo")

ax.set_xlabel("Días")
ax.set_ylabel("Nivel de ansiedad")
ax.set_title("Evolución simulada de Ansiedad y zonas")
ax.legend()
st.pyplot(fig)

# ========== 9.1) Probabilidades de estados (solo Markov) ===========
if modelo == "Cadena de Markov (regímenes discretos)":
    fig2, ax2 = plt.subplots()
    ax2.plot(range(dias), probs_markov[:, 0], label="Prob(BAJA)", linewidth=2)
    ax2.plot(range(dias), probs_markov[:, 1], label="Prob(MEDIA)", linewidth=2)
    ax2.plot(range(dias), probs_markov[:, 2], label="Prob(ALTA)", linewidth=2)
    ax2.set_xlabel("Días")
    ax2.set_ylabel("Probabilidad")
    ax2.set_ylim(0, 1)
    ax2.set_title("Probabilidades de estados (Markov)")
    ax2.legend()
    st.pyplot(fig2)

    st.caption(
        "Las probabilidades dependen de E,S,T,M y Sp. **Mayor apoyo/terapia/medicación** desplazan masa hacia BAJA; **mayor estrés/sensibilidad** la desplazan hacia ALTA."
    )

# ========== 🔟 NOTA FINAL ===========
st.info(
    """
**Este simulador es educativo y no reemplaza atención profesional.  
Si tu ansiedad persiste o se agrava, busca acompañamiento con un especialista en salud mental.  
No te juzgues por los retrocesos: la recuperación es un camino con subidas y bajadas.**
"""
)

# ========== 1️⃣1️⃣ FÓRMULA ODE USADA ===========
with st.expander("Ver detalles de la fórmula matemática (modelo ODE)"):
    st.latex(r"""
    \frac{dA}{dt} = -k_1 A(t) + Sp \cdot k_2 E - Sp(k_3 S + k_4 T + k_5 M)
    """)
    st.write(
        """
        Donde:
        - $A(t)$: nivel de ansiedad en el tiempo
        - $Sp$: sensibilidad personal (ajusta el impacto de factores)
        - $E$: eventos estresantes
        - $S$: apoyo social
        - $T$: terapia psicológica
        - $M$: medicación
        """
    )

# ========== 1️⃣2️⃣ ESPECIFICACIÓN MARKOV (RESUMEN) ===========
with st.expander("Ver detalles del modelo de cadenas de Markov"):
    st.markdown(
        r"""
**Estados**: BAJA, MEDIA, ALTA con valores esperados $\mu_L, \mu_M, \mu_H$.  
**Transición** (tiempo-discreta): $\pi_{t+1} = \pi_t P_t$, donde cada fila de $P_t$ es un softmax de **puntuaciones**:

- Hacia BAJA: $s_L = b_L - w_E e + w_S s + w_T t + w_M m - w_{Sp}(Sp-1)$
- Hacia MEDIA: $s_M = b_M$
- Hacia ALTA: $s_H = b_H + w_E e - w_S s - w_T t - w_M m + w_{Sp}(Sp-1)$

(con **inercia** $\kappa$ sumada al destino igual al estado actual).  
$e,s,t,m$ son $E,S,T,M$ reescalados a $[0,1]$.

**Objetivo**: se reporta el primer día en que la probabilidad acumulada de estar en estados con $\mu \le A_{OBJ}$ alcanza 25%, 50% y 75%.
"""
    )
