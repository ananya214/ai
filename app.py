import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="AI Food Safety Advisor", layout="centered")

st.markdown("""
<style>
.stApp{
background: linear-gradient(135deg,#0f0f0f,#1c1c1c);
color:white;
font-family: 'Segoe UI', sans-serif;
}
.main-title{
text-align:center;
font-size:42px;
font-weight:700;
color:#8b9cff;
}
.subtitle{
text-align:center;
color:#cccccc;
margin-bottom:20px;
}
.safe{
background:rgba(0,255,150,0.12);
padding:12px;
border-radius:10px;
color:#00ffa6;
font-weight:600;
}
.warning{
background:rgba(255,200,0,0.12);
padding:12px;
border-radius:10px;
color:#ffd54a;
font-weight:600;
}
.danger{
background:rgba(255,0,80,0.15);
padding:12px;
border-radius:10px;
color:#ff4d6d;
font-weight:600;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    food_model = load_model("food_detection_model.keras", compile=False)
    spoilage_model = load_model("food_spoilage_model.h5", compile=False)
    return food_model, spoilage_model

food_model, spoilage_model = load_models()

st.markdown('<div class="main-title">🥗 AI Food Safety Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect food type and freshness using AI</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Food Image", type=["jpg","jpeg","png","webp"])

# ---------------- FOOD DETECTION ----------------
def detect_food(img):
    img = img.resize((150,150))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = food_model.predict(img_array, verbose=0)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index])

    classes = ["dal","other","rice","roti"]

    if confidence < 0.60:
        food = "other"
    else:
        food = classes[index]

    return food, confidence

# ---------------- SPOILAGE DETECTION ----------------
def detect_spoilage(img):
    img = img.resize((150,150))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = spoilage_model.predict(img_array, verbose=0)
    prob = float(prediction[0][0])

    if prob > 0.45:
        return "Mold", prob
    else:
        return "Fresh", 1 - prob

# ---------------- SAFETY LOGIC ----------------
def safety_logic(food, hours, env, temperature):

    risk = 0
    explanation = []

    # 🌡️ Temperature logic
    if temperature >= 35:
        risk += 3
        explanation.append("Very high temperature → rapid bacterial growth.")

    elif temperature >= 25:
        risk += 2
        explanation.append("Warm temperature → bacteria grow quickly.")

    elif temperature >= 15:
        risk += 1
        explanation.append("Moderate temperature → some bacterial growth.")

    else:
        explanation.append("Cool temperature → slower bacterial activity.")

    # 🏠 Environment logic
    if env == "Refrigerator":
        risk -= 2
        explanation.append("Refrigeration slows spoilage.")

    elif env == "Winter":
        risk -= 1
        explanation.append("Cold weather helps preserve food.")

    elif env == "Summer":
        risk += 1
        explanation.append("Summer increases spoilage risk.")

    # ⏳ Time logic
    if hours > 10:
        risk += 2
        explanation.append("Food stored too long.")

    elif hours > 6:
        risk += 1
        explanation.append("Food sitting for several hours.")

    # 🍚 Food logic
    if food == "rice":
        risk += 1
        explanation.append("Rice can grow harmful bacteria.")

    # 🎯 Final decision
    if risk <= 0:
        result = "Safe to Eat"
        explanation.append("Conditions are safe.")

    elif risk <= 2:
        result = "Eat Soon"
        explanation.append("Consume soon.")

    else:
        result = "Not Safe to Eat"
        explanation.append("High contamination risk.")

    return result, explanation

# ---------------- MAIN APP ----------------
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, width=320)

    food, food_conf = detect_food(img)
    condition, cond_conf = detect_spoilage(img)

    st.subheader("AI Analysis")
    st.write(f"Detected Food: **{food}**")
    st.progress(food_conf)

    st.write(f"Food Condition: **{condition}**")
    st.progress(cond_conf)

    if condition == "Mold":
        st.markdown('<div class="danger">❌ Mold detected. Do NOT eat.</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="safe">Food appears fresh</div>', unsafe_allow_html=True)

        hours = st.number_input("Hours since cooked", 0, 48, 0)

        env = st.selectbox(
            "Storage Environment",
            ["Season","Winter","Summer","Refrigerator"]
        )

        # 🌡️ NEW FEATURE
        temperature = st.slider("Room temperature (°C)", 0, 50, 25)

        if st.button("Check Food Safety"):

            result, explanation = safety_logic(food, hours, env, temperature)

            if result == "Safe to Eat":
                st.markdown('<div class="safe">🟢 Safe to Eat</div>', unsafe_allow_html=True)

            elif result == "Eat Soon":
                st.markdown('<div class="warning">⚠ Eat Soon</div>', unsafe_allow_html=True)

            else:
                st.markdown('<div class="danger">❌ Not Safe to Eat</div>', unsafe_allow_html=True)

            st.subheader("Safety Explanation")
            for line in explanation:
                st.write("•", line)