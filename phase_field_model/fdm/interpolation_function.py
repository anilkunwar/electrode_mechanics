import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(theta, c, c0=5/7):
    return 1 / (1 + np.exp(-theta * (c - c0)))

def main():
    st.title('Interpolation Function for Composition Dependent Material Properties in BCT Sn and Li₂Sn₅ Phase')
    thetas = []
    chosen_colors = []
    color_map = ['#FF5733', '#33FF57', '#3366FF', '#FF33FF', '#000000']
    c0 = 5/7  # Transition point for Li₂Sn₅
    xf = st.sidebar.slider(f'$x_{math}$', min_value=0, max_value=1.0, value=0.4)
    yf = st.sidebar.slider(f'$y_{math}$', min_value=0, max_value=1.0, value=0.4)
    for i in range(2):
        theta_default = 20 if i == 0 else 40  # Default θ values inspired by β = 20
        theta = st.sidebar.slider(f'$\\theta_{i+1}$', min_value=0, max_value=100, value=theta_default)
        thetas.append(theta)
        color = st.sidebar.color_picker(f'Choose Color for $\\theta_{i+1}$', value=color_map[i])
        chosen_colors.append(color)

    c_values = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (theta, color) in enumerate(zip(thetas, chosen_colors)):
        h_values = sigmoid(theta, c_values, c0)
        ax.plot(c_values, h_values, label=f'$\\theta_{i+1}$ = {theta}', color=color, linewidth=4)

    ax.set_xlabel('$c$', fontsize=25)
    ax.set_ylabel('$h$', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20, width=5.0, size=8)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle='--', linewidth=1.0)
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.text(xf, yf, r'$h = \frac{1}{1 + e^{-\theta (c - c_0)}}$', 
            fontsize=20, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')
    #ax.text(0.1, 0.4, r'$h = \frac{1}{1 + e^{-\theta (c - c_0)}}$', 
    #        fontsize=20, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')

    st.pyplot(fig)
    
    st.write("### Description:")
    st.write("The sigmoidal interpolation function is given by:")
    st.latex(r'h = \frac{1}{1 + e^{-\theta (c - c_0)}}')
    st.write("where \( c \) is the mole fraction of Sn in the Sn to Li₂Sn₅ phase transformation.")
    st.write(f"The constant \( c_0 = {c0:.3f} \) represents the transition point for Li₂Sn₅ (\( c \approx 0.714 \)).")
    st.write("The term \( \theta (c - c_0) \) is the weighted input to the sigmoidal function.")
    st.write("The constant \( \theta > 0 \) controls the steepness of the sigmoidal function for a given mole fraction \( c \).")
    st.write("The mole fraction of Sn is the input feature to the sigmoidal function, with \( c \approx 0.95 \) for Sn and \( c \approx 0.714 \) for Li₂Sn₅.")

    dense_c_values = np.concatenate([np.linspace(0, 0.4, 5),
                                     np.linspace(0.4, 0.6, 11),
                                     np.linspace(0.6, 1, 5)])
    c_values = np.unique(dense_c_values)

    h_values = {'c': c_values}
    for i, theta in enumerate(thetas):
        col_name = f'h({theta},c)'
        h_values[col_name] = sigmoid(theta, c_values, c0)

    df = pd.DataFrame(h_values)
    df.set_index('c', inplace=True)

    st.write("### Table of \( h(\theta, c) \) for given \( c \)")
    st.write(df)
    
    csv = df.to_csv().encode('utf-8')
    st.download_button(label='Download CSV', data=csv, file_name='interpolation_function.csv', mime='text/csv')
    
    st.write("### How to Cite This Work:")
    st.markdown("""
    If you find this work useful, please cite the following paper:

    ```
     A. Kunwar and N. Moelans, Phase field modeling of elastochemical effects at the Sn anode of Lithium-ion batteries (Working Title and May be Updated), 2025 (Work in Progress)
    ```
    """)

if __name__ == '__main__':
    main()
