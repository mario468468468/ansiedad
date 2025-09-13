# filename: dashboard_ansiedad.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Simulador de Ansiedad", layout="centered")
np.set_printoptions(precision=4, suppress=True)

# ---------------- UI mínima ----------------
st.title("🧠 Simulador de Ansiedad")
st.caption("Educativo. No reemplaza diagnóstico profesional.")

with st.sidebar:
    st.header("Entradas")
    A0 = st.slider("Ansiedad inicial", 0, 100, 70)
    E  = st.slider("Estrés (E)", 0, 100, 60)
    S  = st.slider("Apoyo social (S)", 0, 100, 50)
    Tb = st.slider("Terapia (T)", 0, 100, 40)
    med = st.radio("¿Medicación?", ["No","Sí"])
    Mb = st.slider("Medicación (M)", 0, 100, 30) if med=="Sí" else 0
    # Sensibilidad personal compacta
    Sp = st.slider("Sensibilidad personal (Sp)", 0.5, 2.0, 1.0, 0.05,
                   help=">1 amplifica el impacto de E; <1 amortigua.")
    # Umbrales
    A_OBJ  = st.slider("Objetivo funcional", 0, 100, 40)
    A_BAJA = st.slider("Umbral BAJO", 0, 100, 40)
    A_MEDIA= st.slider("Umbral MEDIO", 0, 100, 60)

    dias = 90
    if st.button("Analizar"):
        st.session_state.run = True

st.write("""
Compara un modelo ODE **realista con rebotes** con un modelo **Markov** interno
(regímenes BAJO/MEDIO/ALTO). Markov se calcula automáticamente a partir de E, S, T, M y Sp.
""")

# Coeficientes fijos
k1, k2, k3, k4, k5 = 0.05, 0.04, 0.03, 0.02, 0.05

def _saturacion(x, max_ef=100, escala=30):
    return max_ef * (1 - np.exp(-x/escala))

def simular_realista(A0, E, S, T, M, Sp, dias, rebotes=True):
    A = np.zeros(dias); A[0] = A0
    b = 0.07 + (0.15 - Sp*0.04)
    w = 0.48 + (E/500)
    for t in range(1, dias):
        S_eff = _saturacion(S, max_ef=100, escala=40)
        M_eff = _saturacion(M, max_ef=100, escala=30)
        Sp_t = Sp + 0.15 if (t>10 and np.mean(A[max(0,t-10):t])>70) else Sp
        dA_dt = -k1*A[t-1] + Sp_t*k2*E - Sp_t*(k3*S_eff + k4*T + k5*M_eff)
        base = A[t-1] + dA_dt
        if rebotes:
            osc = np.exp(-b*t)*np.cos(w*t)*(0.18*A0)
            ruido = np.random.normal(0, 1.3)
        else:
            osc = 0; ruido = 0
        A[t] = max(0, base + osc + ruido)
    return A

def _softmax(x):
    x = x - np.max(x); ex = np.exp(x)
    return ex/np.sum(ex)

# ---- Markov 3 estados (Bajo, Medio, Alto) ----
MU = np.array([30.0, 55.0, 80.0])
INERCIA = 1.0
W = dict(E=1.0, S=1.0, T=0.8, M=1.2, Sp=1.0)
B = dict(L=0.0, M=0.0, H=0.0)

def _fila_transicion(curr_idx, E,S,T,M,Sp):
    scale = 1/100.0
    e,s,t,m = E*scale, S*scale, T*scale, M*scale
    sL = B["L"] + (-W["E"]*e + W["S"]*s + W["T"]*t + W["M"]*m - W["Sp"]*(Sp-1))
    sM = B["M"]
    sH = B["H"] + (+W["E"]*e - W["S"]*s - W["T"]*t - W["M"]*m + W["Sp"]*(Sp-1))
    if   curr_idx==0: sL += INERCIA
    elif curr_idx==1: sM += INERCIA
    else:             sH += INERCIA
    return _softmax(np.array([sL,sM,sH], float))

def _init_dist(A0, A_BAJA, A_MEDIA):
    if A0 <= A_BAJA: return np.array([1.0,0.0,0.0])
    if A0 <= A_MEDIA:return np.array([0.0,1.0,0.0])
    return np.array([0.0,0.0,1.0])

def sim_markov(A0,E,S,T,M,Sp,dias):
    dist = _init_dist(A0, A_BAJA, A_MEDIA)
    probs = np.zeros((dias,3)); expA = np.zeros(dias)
    for t in range(dias):
        probs[t] = dist
        expA[t] = MU @ dist
        P = np.vstack([
            _fila_transicion(0,E,S,T,M,Sp),
            _fila_transicion(1,E,S,T,M,Sp),
            _fila_transicion(2,E,S,T,M,Sp),
        ])
        dist = dist @ P
    return probs, expA

def _dia_umbral(arr, umbral):
    for i,v in enumerate(arr):
        if v <= umbral: return i
    return None

def _dia_prob_objetivo(probs, MU, A_OBJ, q=0.5):
    goal = np.where(MU <= A_OBJ)[0]
    if len(goal)==0: return None
    curva = probs[:, goal].sum(axis=1)
    for t,p in enumerate(curva):
        if p>=q: return t
    return None

# ---------------- Run ----------------
if st.session_state.get("run"):
    # ODE principal
    A = simular_realista(A0,E,S,Tb,Mb,Sp,dias,rebotes=True)
    d_baja = _dia_umbral(A, A_BAJA)
    d_media= _dia_umbral(A, A_MEDIA)
    d_obj  = _dia_umbral(A, A_OBJ)

    st.subheader("⏳ Tiempo estimado (ODE realista)")
    if d_obj is not None:
        variab = int(5 + abs(Sp-1)*6 + abs(E-50)/20)
        st.write(f"Objetivo ({A_OBJ}) entre **{max(0,d_obj-variab)}** y **{min(dias-1,d_obj+variab)}** días.")
    else:
        st.write("Sin cruce del objetivo en 90 días.")
    if d_baja is not None: st.write(f"Zona BAJA ~ día {d_baja}.")
    if d_media is not None:st.write(f"Zona MEDIA ~ día {d_media}.")

    # Markov
    probs, expA = sim_markov(A0,E,S,Tb,Mb,Sp,dias)
    d_p50 = _dia_prob_objetivo(probs, MU, A_OBJ, q=0.50)
    st.subheader("🎯 Probabilidad de alcanzar objetivo (Markov)")
    st.write(f"Día mediano (50%): **{d_p50}**." if d_p50 is not None else
             "No se alcanza 50% ≤ objetivo en 90 días.")

    # Gráfico ODE vs Markov
    st.subheader("Evolución simulada")
    fig, ax = plt.subplots()
    ax.plot(range(dias), A, label="ODE realista", linewidth=2)
    ax.plot(range(dias), expA, label="E[A_t] Markov", linewidth=2)
    ax.axhline(y=A_BAJA, linestyle="--", label="Umbral BAJO")
    ax.axhline(y=A_MEDIA, linestyle="--", label="Umbral MEDIO")
    ax.axhline(y=A_OBJ, linestyle=":",  label="Objetivo")
    ax.set_xlabel("Días"); ax.set_ylabel("Ansiedad"); ax.legend()
    st.pyplot(fig)

    # Probabilidades por estado
    st.subheader("Probabilidades de estados (Markov)")
    fig2, ax2 = plt.subplots()
    ax2.plot(range(dias), probs[:,0], label="Prob(BAJA)", linewidth=2)
    ax2.plot(range(dias), probs[:,1], label="Prob(MEDIA)", linewidth=2)
    ax2.plot(range(dias), probs[:,2], label="Prob(ALTA)",  linewidth=2)
    ax2.set_xlabel("Días"); ax2.set_ylabel("Probabilidad"); ax2.set_ylim(0,1); ax2.legend()
    st.pyplot(fig2)

    # --------- Métricas para interpretación y recomendaciones ----------
    def _volatilidad(arr):
        return float(np.std(np.diff(arr))) if len(arr) > 1 else 0.0
    vol = _volatilidad(A)
    trend = float(np.mean(A[-10:]) - np.mean(A[:10])) if len(A) >= 20 else 0.0
    p_high_mean = float(probs[:, 2].mean())
    p_low_last30 = float(probs[-30:, 0].mean()) if dias >= 30 else float(probs[:,0].mean())

    def _score(E,S,T,M):
        _, expA_cf = sim_markov(A0,E,S,T,M,Sp,dias)
        return float(np.mean(expA_cf[-30:])) if dias >= 30 else float(np.mean(expA_cf))

    base_score = _score(E,S,Tb,Mb)
    candidates = {
        "↓ Estrés −15":  (max(0, E-15), S, Tb, Mb),
        "↑ Apoyo +15":   (E, min(100, S+15), Tb, Mb),
        "↑ Terapia +15": (E, S, min(100, Tb+15), Mb),
        "↗ Meds +10":    (E, S, Tb, min(100, Mb+10)),
    }
    deltas = {k: base_score - _score(*v) for k, v in candidates.items()}
    best2 = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)[:2]
    best_lever, best_gain = best2[0][0], best2[0][1]

    def _day50(E,S,T,M):
        p_cf, _ = sim_markov(A0,E,S,T,M,Sp,dias)
        goal = np.where(MU <= A_OBJ)[0]
        if len(goal)==0: return None
        curve = p_cf[:, goal].sum(axis=1)
        for t,p in enumerate(curve):
            if p>=0.5: return t
        return None

    base_d50 = d_p50
    lever_params = candidates[best_lever]
    lever_d50 = _day50(*lever_params)

    # -------------------- INTERPRETACIÓN --------------------
    st.subheader("🧩 Interpretación de tus resultados")

    interpret = []

    # Perfil global
    if d_obj is None:
        perfil = "sin-cruce"
        interpret.append("La curva ODE no alcanza el objetivo en 90 días: patrón de **riesgo de cronificación**.")
    elif d_obj < 30 and vol < 3:
        perfil = "mejora-rápida-estable"
        interpret.append("Cruce temprano y estable: **recuperación rápida** con baja oscilación.")
    elif d_obj < 30 and vol >= 3:
        perfil = "mejora-rápida-volátil"
        interpret.append("Cruce temprano con oscilación alta: **mejora frágil** con riesgo de recaídas.")
    elif d_obj > 60 and p_high_mean > 0.4:
        perfil = "crónico-alto"
        interpret.append("Cruce tardío y alta permanencia en ALTA: **ansiedad rígida** con fuerte inercia.")
    else:
        perfil = "intermedio"
        interpret.append("Cruce intermedio con progresión gradual: **mejora lenta pero sostenida**.")

    # Interacciones de factores
    if E > 70 and S < 40 and Sp > 1.2:
        interpret.append("Estrés alto + apoyo bajo + sensibilidad elevada: combinación que **perpetúa** el ciclo ansioso.")
    if Tb < 30 and perfil in ["crónico-alto", "sin-cruce"]:
        interpret.append("Terapia actual baja frente a patrón de cronificación: el **enfoque** puede ser insuficiente.")
    if Mb == 0 and perfil == "crónico-alto":
        interpret.append("Sin medicación en patrón rígido: podría faltar **modulación** biológica del síntoma.")
    if p_low_last30 < 0.35 and d_obj is not None and d_obj > 45:
        interpret.append("Baja permanencia en BAJA al final del horizonte: falta **mantenimiento** y consolidación.")
    if trend > 0 and d_obj is not None and d_obj < 30:
        interpret.append("Ligero repunte reciente pese a cruce temprano: riesgo de **exceso de carga** al mejorar.")

    for msg in interpret:
        st.markdown(f"- {msg}")

    # -------------------- RECOMENDACIONES (tono empático) --------------------
    st.subheader("🤝 Recomendaciones personalizadas")

    recs = []

    # Plan según perfil
    if perfil == "mejora-rápida-estable":
        recs.append("Vas bien. Mantén lo que ya funciona y protege tus rutinas de sueño y descanso. La estabilidad es tu mejor aliada.")
    if perfil == "mejora-rápida-volátil":
        recs.append("La mejora está, pero es frágil. Vale priorizar estabilidad: agenda horarios fijos, pausa breve después de picos y una rutina simple de fin de día.")
    if perfil == "intermedio":
        recs.append("El avance es real aunque lento. Suma un ajuste pequeño y sostenible esta semana en vez de muchos cambios a la vez.")
    if perfil == "sin-cruce":
        recs.append("No ver cruce aún puede desanimar. Tiene sentido probar un giro claro: reducir una carga concreta o aumentar el soporte profesional las próximas 2–4 semanas.")
    if perfil == "crónico-alto":
        recs.append("Cuando la ansiedad está rígida, combinar dos palancas suele abrir camino. Empecemos por la más efectiva y añade una segunda en paralelo.")

    # Palancas comparativas con tono cercano
    recs.append(f"La palanca que más te ayuda ahora es **{best_lever}** (mejora aproximada {best_gain:.1f} puntos en 30 días).")
    if len(best2) > 1 and best2[1][1] > 0.5:
        recs.append(f"También suma **{best2[1][0]}** (≈ {best2[1][1]:.1f} puntos).")
    if (base_d50 is not None) and (lever_d50 is not None) and (lever_d50 < base_d50):
        recs.append(f"Aplicando {best_lever}, el día mediano para estar en tu objetivo podría pasar de **{base_d50}** a **{lever_d50}**. Es un cambio valioso.")

    # Casos específicos
    if p_high_mean >= 0.40:
        recs.append("Con tanta permanencia en ALTA, es razonable recortar una fuente de estrés concreta por 2 semanas y cuidar el sueño como prioridad.")
    if vol > 4.0:
        recs.append("La oscilación alta cansa. Prueba un amortiguador diario breve: 5 minutos de respiración 4-4-6 o una caminata corta después del pico.")
    if S < 40:
        recs.append("El apoyo importa. Elige 2 personas de confianza y agenda contacto breve esta semana. Pide algo concreto y fácil de cumplir.")
    if E > 70:
        recs.append("El estrés está pesado. Quita o delega una tarea específica. Una menos ya reduce carga y te da espacio para recuperarte.")
    if Tb < 30 and (d_obj is not None and d_obj > 45):
        recs.append("Si la terapia es baja y el cruce tarda, subir estructura o frecuencia puede darte el empujón que falta.")
    if Sp > 1.3:
        recs.append("Con alta sensibilidad, menos es más: una cosa a la vez, sin multitarea, y un cierre de día predecible ayuda bastante.")
    if Mb == 0 and perfil == "crónico-alto":
        recs.append("Si te lo permite tu equipo de salud, conversa sobre opciones farmacológicas. A veces bajar ‘ruido’ fisiológico facilita el resto.")

    for r in recs:
        st.markdown(f"- {r}")

    st.caption("Estas sugerencias son informativas y buscan acompañarte. Ajusta siempre con tu profesional de salud.")
else:
    st.info("Configura entradas en la barra lateral y pulsa Analizar.")
