import streamlit as st
import pandas as pd
import pickle

# ──────────────────────────────────────────────────────────────────────────────
# Configuramos la página
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción % de Aforo y Taquilla",
    layout="centered"
)

# ──────────────────────────────────────────────────────────────────────────────
# Cargamos los modelos de predicción obtenidos
# ──────────────────────────────────────────────────────────────────────────────
def _load_models():
    rf_clf = pickle.load(open('rf_pipeline.pkl', 'rb'))
    rf_reg = pickle.load(open('best_model_reg.pkl', 'rb'))
    return rf_clf, rf_reg

load_models = st.cache_resource(_load_models)
clf, reg = load_models()

# ──────────────────────────────────────────────────────────────────────────────
# Título de la página
# ──────────────────────────────────────────────────────────────────────────────
st.title("🎟️ Predicción % de Aforo y Taquilla")

st.markdown("""
Introduce los datos del concierto a predecir.  
Los datos de Entradas acumuladas en las diferentes semanas deben cumplir:  
`4sem ≥ 8sem ≥ 12sem ≥ 16sem`
""")

# ──────────────────────────────────────────────────────────────────────────────
# Variables predictoras
# ──────────────────────────────────────────────────────────────────────────────
Aforo              = st.number_input("Aforo total", min_value=0, value=1000)
NConcCompProx      = st.number_input("N° conciertos competencia ±3d", min_value=0, value=0)
ConcProxFil        = st.number_input("N° conciertos propios ±7d", min_value=0, value=0)
Num_Participativos = st.number_input("Num.Participativos", min_value=0, value=0)
Antelacion_Media   = st.number_input("Antelación media (días)", min_value=0, value=0)
Entradas16         = st.number_input("EntradasAcum_16sem", min_value=0, value=0)
Entradas12         = st.number_input("EntradasAcum_12sem", min_value=0, value=0)
Entradas8          = st.number_input("EntradasAcum_8sem",  min_value=0, value=0)
Entradas4          = st.number_input("EntradasAcum_4sem",  min_value=0, value=0)

# ──────────────────────────────────────────────────────────────────────────────
# Mes del evento (1-12)
# ──────────────────────────────────────────────────────────────────────────────
Mes_evento = st.selectbox(
    "Mes del evento",
    options=list(range(1,13)),
    format_func=lambda x: f"{x:02d}",  # opcional: muestra siempre dos dígitos
    index=0
)

tipo_list = clf.named_steps['pre']\
    .named_transformers_['cat']\
    .categories_[0].tolist()
TipoConcierto = st.selectbox("TipoConcierto", options=tipo_list)
DiaSemana     = st.selectbox("DiaSemana", options=[
    'Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'
])
HoraConcierto = st.text_input("HoraConcierto (HH:MM:SS)", "19:30:00")

# ──────────────────────────────────────────────────────────────────────────────
# Botón de predicción
# ──────────────────────────────────────────────────────────────────────────────
if st.button("Calcular predicciones"):

    # Validar coherencia acumulados
    if not (Entradas4 >= Entradas8 >= Entradas12 >= Entradas16):
        st.error("⚠️ Error: deben cumplirse 4sem ≥ 8sem ≥ 12sem ≥ 16sem.")
    else:
        data = {
            'Aforo':              Aforo,
            'NConcCompProx':      NConcCompProx,
            'ConcProxFil':        ConcProxFil,
            'Num.Participativos': Num_Participativos,
            'Antelacion_Media':   Antelacion_Media,
            'EntradasAcum_16sem': Entradas16,
            'EntradasAcum_12sem': Entradas12,
            'EntradasAcum_8sem':  Entradas8,
            'EntradasAcum_4sem':  Entradas4,
            'Mes_evento':         Mes_evento,        
            'TipoConcierto':      TipoConcierto,
            'DiaSemana':          DiaSemana,
            'HoraConcierto':      HoraConcierto
        }
        X_new = pd.DataFrame([data])

        # Predicción de Aforo
        aforo_pred   = clf.predict(X_new)[0]
        aforo_proba  = dict(zip(
            clf.named_steps['clf'].classes_,
            clf.predict_proba(X_new)[0]
        ))

        # Predicción de Taquilla
        taquilla_pred = reg.predict(X_new)[0]

        # ────────────────────────────────────────────────────────────────
        # Mostramos los resultados
        # ────────────────────────────────────────────────────────────────
        st.markdown("## 📊 Resultados")
        # Predicción % de Aforo
        st.markdown(
            f"<h1 style='color:teal; text-align:center;'>% de Aforo: {aforo_pred}</h1>",
            unsafe_allow_html=True
        )
        # Probabilidades de cada clase
        st.markdown("<h4>Probabilidades:</h4>", unsafe_allow_html=True)
        for cls, p in aforo_proba.items():
            st.markdown(f"- **{cls}**: {p:.2%}")

        # Predicción Taquilla
        st.markdown(
            f"<h1 style='color:crimson; text-align:center;'>Taquilla estimada: €{taquilla_pred:,.2f}</h1>",
            unsafe_allow_html=True
        )

