import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
c1, c2 = 0.71, 0.95
c0 = (c1 + c2) / 2
target = 0.98  # h(c1)=0.01, h(c2)=0.99 → 2h - 1 = ±0.98

# Functions
def auto_theta_sigmoid(c1, c2):
    return np.log(target / (1 - target)) / (c2 - c1)

def auto_theta_arctan(c1, c2):
    return np.tan(np.pi * (target - 0.5)) / (c2 - c0)

def auto_theta_tanh(c1, c2):
    return np.arctanh(target) / (c2 - c0)

def auto_theta_logsigmoid(c1, c2):
    # Based on numerical estimation for matching sigmoid steepness
    return auto_theta_sigmoid(c1, c2)

def sigmoid_transition(c, theta):
    return 1 / (1 + np.exp(-theta * (c - c0)))

def arctan_transition(c, theta):
    return (np.arctan(theta * (c - c0)) / np.pi) + 0.5

def tanh_transition(c, theta):
    return 0.5 * np.tanh(theta * (c - c0)) + 0.5

def logsigmoid_transition(c, theta):
    e_term = np.log(1 + np.exp(theta * (c - c0)))
    normalizer = np.log(1 + np.exp(theta * (c2 - c0)))
    return e_term / normalizer

# Streamlit App
def main():
    st.title('Smooth Transition Function Explorer')
    st.write("Choose between **Sigmoid**, **Arctangent**, **Tanh**, and **Log-sigmoid** interpolation models.")

    # Model selection
    model = st.selectbox("Select transition model:", ["Sigmoid", "Arctangent", "Tanh", "Log-sigmoid"])

    if model == "Sigmoid":
        theta_auto = auto_theta_sigmoid(c1, c2)
        formula = r"h(c) = \frac{1}{1 + e^{-\theta(c - c_0)}}"
        h_func = sigmoid_transition
    elif model == "Arctangent":
        theta_auto = auto_theta_arctan(c1, c2)
        formula = r"h(c) = \frac{1}{\pi} \arctan(\theta (c - c_0)) + \frac{1}{2}"
        h_func = arctan_transition
    elif model == "Tanh":
        theta_auto = auto_theta_tanh(c1, c2)
        formula = r"h(c) = \frac{1}{2} \tanh(\theta (c - c_0)) + \frac{1}{2}"
        h_func = tanh_transition
    elif model == "Log-sigmoid":
        theta_auto = auto_theta_logsigmoid(c1, c2)
        formula = r"h(c) = \frac{\log(1 + e^{\theta (c - c_0)})}{\log(1 + e^{\theta (c_{\mathrm{max}} - c_0)})}"
        h_func = logsigmoid_transition

    st.latex(formula)
    st.write(f"Automatically calculated $\\theta$: **{theta_auto:.4f}**")

    # User-adjustable theta
    theta = st.slider("Adjust θ (steepness)", min_value=0.1, max_value=float(theta_auto * 2),
                      value=float(theta_auto), step=0.01)

    # Sidebar plot control
    xf = st.sidebar.slider('x-text position', 0.0, 1.0, 0.5)
    yf = st.sidebar.slider('y-text position', 0.0, 1.0, 0.4)
    curve_color = st.sidebar.color_picker("Pick line color", "#1f77b4")

    # Generate and plot
    c_vals = np.linspace(0, 1, 400)
    h_vals = h_func(c_vals, theta)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(c_vals, h_vals, color=curve_color, linewidth=4,
            label=fr"$\theta$ = {theta:.2f}")

    ax.set_xlabel('$c$', fontsize=25)
    ax.set_ylabel('$h$', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20, width=5.0, size=8)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle='--', linewidth=1.0)

    for spine in ax.spines.values():
        spine.set_linewidth(4)

    ax.text(xf, yf, formula, fontsize=20, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right')

    st.pyplot(fig)

    # Table display
    st.write("### Table of \( h(c) \) for selected \( c \) values:")
    sample_c = np.round(np.linspace(0, 1, 21), 3)
    h_sample = h_func(sample_c, theta)
    df = pd.DataFrame({'c': sample_c, 'h(c)': np.round(h_sample, 5)}).set_index('c')
    st.dataframe(df)

    csv = df.to_csv().encode('utf-8')
    st.download_button(label="Download CSV", data=csv,
                       file_name=f"{model.lower()}_interpolation.csv", mime="text/csv")

if __name__ == '__main__':
    main()
