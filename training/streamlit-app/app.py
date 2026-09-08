import streamlit as st
from PIL import Image

from model_helper import predict, class_names


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AutoGuard AI",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚗 AutoGuard AI")

st.subheader("Vehicle Damage Detection & Classification")

st.write(
    "Upload a vehicle image and let our AI model identify "
    "the type and location of damage."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Model:** ResNet50")
    st.write("**Task:** Vehicle Damage Classification")
    st.write("**Classes:** 6")
    st.write("**Test Accuracy:** 77%")

    st.divider()

    st.write("### Classification Classes")

    for name in class_names:
        st.write("•", name)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📤 Upload Vehicle Image")

uploaded_file = st.file_uploader(
    "Choose a vehicle image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# IF IMAGE IS UPLOADED
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    # ========================================================
    # IMAGE + RESULT COLUMNS
    # ========================================================

    image_col, result_col = st.columns(
        [1.2, 1],
        gap="large"
    )


    # ========================================================
    # IMAGE
    # ========================================================

    with image_col:

        st.subheader("🖼️ Vehicle Image")

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"File: {uploaded_file.name}"
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    with result_col:

        st.subheader("🤖 AI Analysis")

        if st.button(
            "🔍 Analyze Vehicle",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing vehicle..."
            ):

                predicted_name, confidence, probabilities = predict(
                    image
                )


            # =================================================
            # MAIN RESULT
            # =================================================

            st.success(
                "Analysis Complete"
            )

            st.metric(
                label="Predicted Class",
                value=predicted_name
            )

            st.metric(
                label="Confidence",
                value=f"{confidence * 100:.2f}%"
            )


            # =================================================
            # DAMAGE INFORMATION
            # =================================================

            if "Normal" in predicted_name:

                st.info(
                    "✅ The model predicts that this vehicle "
                    "area appears to be normal."
                )

            else:

                st.warning(
                    "⚠️ Damage has been detected in this "
                    "vehicle area."
                )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.divider()

    st.header("📊 Detailed Prediction")

    st.write(
        "Probability assigned by the model to each class:"
    )


    # Combine names and probabilities

    results = list(
        zip(
            class_names,
            probabilities
        )
    )


    # Sort highest probability first

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    # Display probabilities

    for name, probability in results:

        percentage = probability * 100

        st.write(
            f"**{name} — {percentage:.2f}%**"
        )

        st.progress(
            float(probability)
        )


# ============================================================
# NO IMAGE
# ============================================================

else:

    st.info(
        "👆 Upload a vehicle image above to begin the analysis."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    ### 🚗 AutoGuard AI

    **AI-powered vehicle damage classification**

    Designed & Developed by **Aman Lamje **

    *Machine Learning • Deep Learning • Artificial Intelligence*

    `PyTorch` • `ResNet50` • `Streamlit` • `FastAPI`
    """
)

st.caption(
    "Vehicle Damage Prediction Project"
)