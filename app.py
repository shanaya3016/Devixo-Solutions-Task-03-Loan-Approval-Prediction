import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💸", layout="centered")

# ---------- Styling ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=DM+Sans:wght@400;500;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(160deg, #F3EFFF 0%, #FFF1F0 55%, #E6FBF8 100%); }
.block-container { padding-top: 1.5rem; max-width: 880px; }
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(120deg, #3B1F9E 0%, #6C4CF1 45%, #19C3B1 100%);
    border-radius: 28px; padding: 38px 36px; color: #fff; margin-bottom: 26px;
    box-shadow: 0 18px 40px rgba(108, 76, 241, .28); position: relative; overflow: hidden;
}
.hero h1 { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 2.3rem; margin: 0 0 6px 0; color: #fff; }
.hero p { font-size: 1.05rem; margin: 0; opacity: .92; max-width: 520px; }
.hero .coin { position: absolute; right: -30px; top: -30px; width: 170px; height: 170px;
    border-radius: 50%; background: #FFC145; opacity: .9; }
.hero .coin2 { position: absolute; right: 70px; bottom: -50px; width: 110px; height: 110px;
    border-radius: 50%; background: #FF6B6B; opacity: .85; }
.hero .badge { position: absolute; right: 38px; top: 38px; font-size: 3.2rem; }

.chip { display: inline-block; font-family: 'Sora', sans-serif; font-weight: 700; font-size: .95rem;
    padding: 6px 14px; border-radius: 999px; color: #fff; margin: 4px 0 10px 0; }
.chip.violet { background: #6C4CF1; }
.chip.coral  { background: #FF6B6B; }
.chip.aqua   { background: #12A89A; }

[data-testid="stForm"] { background: #ffffffcc; border: 2px solid #E4DCFF; border-radius: 24px; padding: 22px 24px; }
label, .stNumberInput label p, .stSlider label p { color: #2A2358 !important; font-weight: 700 !important; }

div.stButton, div[data-testid="stFormSubmitButton"] { width: 100% !important; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    width: 100% !important; min-height: 54px; white-space: nowrap; border: 0; border-radius: 16px; padding: 14px 24px; font-family: 'Sora', sans-serif;
    font-weight: 700; font-size: 1.1rem; color: #fff;
    background: linear-gradient(90deg, #FF6B6B, #FFA94D);
    box-shadow: 0 10px 22px rgba(255, 107, 107, .35);
}
div[data-testid="stFormSubmitButton"] > button:hover { filter: brightness(1.06); color: #fff; }

.result { border-radius: 26px; padding: 30px 32px; color: #fff; margin-top: 22px;
    box-shadow: 0 16px 36px rgba(0,0,0,.16); }
.result.ok  { background: linear-gradient(120deg, #0CA678, #38D9A9); }
.result.bad { background: linear-gradient(120deg, #E8456B, #FF8A5C); }
.result h2 { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 2rem; margin: 0 0 4px 0; color: #fff; }
.result p { margin: 0 0 16px 0; font-size: 1.02rem; opacity: .95; }
.meter { position: relative; height: 16px; background: rgba(255,255,255,.35); border-radius: 999px; }
.meter .fill { height: 100%; background: #fff; border-radius: 999px; }
.meter .mid { position: absolute; left: 50%; top: -5px; width: 3px; height: 26px; background: #14123B; border-radius: 2px; }
.meter-label { display: flex; justify-content: space-between; font-size: .85rem; margin-top: 8px; opacity: .95; }

.stats { display: flex; gap: 14px; margin-top: 18px; flex-wrap: wrap; }
.stat { flex: 1; min-width: 180px; background: #fff; border-radius: 20px; padding: 16px 18px;
    border-top: 6px solid var(--c); box-shadow: 0 8px 20px rgba(60, 40, 140, .10); }
.stat .v { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.5rem; color: #14123B; }
.stat .k { font-size: .9rem; color: #5B5683; }
.note { color: #5B5683; font-size: .88rem; margin-top: 22px; text-align: center; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    return (
        joblib.load("models/best_model.joblib"),
        joblib.load("models/selected_features.joblib"),
    )


model, features = load_artifacts()

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
        <div class="coin"></div><div class="coin2"></div>
        <h1>Loan Approval Predictor</h1>
        <p>Fill in the applicant's details and get an instant approve or reject decision
        from a tuned Gradient Boosting model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Form ----------
with st.form("loan_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<span class="chip violet">Applicant</span>', unsafe_allow_html=True)
        dependents = st.number_input("Number of dependents", 0, 5, 2)
        income = st.number_input("Annual income", 100000, 10000000, 5000000, step=100000)
        cibil = st.slider("CIBIL score", 300, 900, 650)

    with c2:
        st.markdown('<span class="chip coral">Loan</span>', unsafe_allow_html=True)
        loan_amount = st.number_input("Loan amount", 100000, 40000000, 15000000, step=100000)
        loan_term = st.number_input("Loan term (years)", 2, 20, 10)

    with c3:
        st.markdown('<span class="chip aqua">Assets</span>', unsafe_allow_html=True)
        residential = st.number_input("Residential assets", 0, 30000000, 5000000, step=100000)
        commercial = st.number_input("Commercial assets", 0, 20000000, 5000000, step=100000)
        luxury = st.number_input("Luxury assets", 0, 40000000, 10000000, step=100000)
        bank = st.number_input("Bank assets", 0, 15000000, 5000000, step=100000)

    submitted = st.form_submit_button("Check my loan", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    total_assets = residential + commercial + luxury + bank
    loan_to_income = loan_amount / income
    assets_to_loan = total_assets / loan_amount
    monthly_payment = loan_amount / (loan_term * 12)

    row = {
        "no_of_dependents": dependents,
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil,
        "luxury_assets_value": luxury,
        "total_assets": total_assets,
        "loan_to_income": loan_to_income,
        "assets_to_loan": assets_to_loan,
        "monthly_payment": monthly_payment,
    }
    X_new = pd.DataFrame([row])[features]

    pred = model.predict(X_new)[0]
    prob = float(model.predict_proba(X_new)[0][1])

    if pred == 1:
        cls, title = "ok", "Loan approved 🎉"
        msg = "This profile looks strong. The model is confident the loan would be granted."
    else:
        cls, title = "bad", "Loan rejected"
        msg = "This profile falls short. A higher CIBIL score is the biggest lever for approval."

    st.markdown(
        f"""
        <div class="result {cls}">
            <h2>{title}</h2>
            <p>{msg}</p>
            <div class="meter"><div class="fill" style="width:{prob*100:.1f}%"></div><div class="mid"></div></div>
            <div class="meter-label"><span>Reject</span><span>Approval probability: <b>{prob:.1%}</b></span><span>Approve</span></div>
        </div>
        <div class="stats">
            <div class="stat" style="--c:#6C4CF1"><div class="v">{cibil}</div><div class="k">CIBIL score</div></div>
            <div class="stat" style="--c:#FF6B6B"><div class="v">{loan_to_income:.2f}x</div><div class="k">Loan to income</div></div>
            <div class="stat" style="--c:#19C3B1"><div class="v">{assets_to_loan:.2f}x</div><div class="k">Assets to loan</div></div>
            <div class="stat" style="--c:#FFC145"><div class="v">{monthly_payment:,.0f}</div><div class="k">Monthly payment</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if pred == 1:
        st.balloons()

st.markdown(
    '<div class="note">Built with scikit-learn and Streamlit. Trained on the Kaggle Loan Approval dataset. '
    'For learning purposes only.</div>',
    unsafe_allow_html=True,
)