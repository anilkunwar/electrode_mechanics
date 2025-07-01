import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Original sigmoid
def sigmoid(theta, c, c0=5/7):
    return 1 / (1 + np.exp(-theta * (c - c0)))

# Smooth transition with auto-calculated θ
def smooth_transition(c, c1=0.71, c2=0.95):
    c0 = (c1 + c2) / 2
    theta = np.log(99) / (c0 - c1)  # ensures h(c1)=0.01, h(c2)=0.99
    return 1 / (1 + np.exp(-theta * (c - c0)))

def main():
    st.title('Interpolation Function for Composition Dependent Material Properties')
    st.write("Use this tool to compare **standard sigmoid** and a **smooth transition function** between Sn and Li₂Sn₅ phases.")

    # Choose function type
    func_type = st.sidebar.radio("Select Interpolation Function Type:", ['Original Sigmoid', 'Smooth Transition'])

    # Plot text placement control
    xf = st.sidebar.slider('$x_m$', min_value=0.0, max_value=1.0, value=0.4)
    yf = st.sidebar.slider('$y_m$', min_value=0.0, max_value=1.0, value=0.4)

    thetas = []
    chosen_colors = []
    color_map = ['#FF5733', '#33FF57', '#3366FF', '#FF33FF', '#000000']

    if func_type == 'Original Sigmoid':
        for i in range(2):
            theta_default = 20 if i == 0 else 40
            theta = st.sidebar.slider(f'$\\theta_{i+1}$', min_value=0, max_value=100, value=theta_default)
            thetas.append(theta)
            color = st.sidebar.color_picker(f'Choose Color for $\\theta_{i+1}$', value=color_map[i])
            chosen_colors.append(color)

    c_values = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 5))

    if func_type == 'Original Sigmoid':
        for i, (theta, color) in enumerate(zip(thetas, chosen_colors)):
            h_values = sigmoid(theta, c_values)
            ax.plot(c_values, h_values, label=f'$\\theta_{i+1}$ = {theta}', color=color, linewidth=4)
    else:
        h_values = smooth_transition(c_values)
        ax.plot(c_values, h_values, label='Smooth Transition', color='gray', linestyle='--', linewidth=4)

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

    st.pyplot(fig)

    # Table of h values
    st.write("### Table of \( h(\theta, c) \) for selected \( c \)")

    dense_c_values = np.concatenate([np.linspace(0, 0.4, 5),
                                     np.linspace(0.4, 0.6, 11),
                                     np.linspace(0.6, 1, 5)])
    c_values = np.unique(dense_c_values)

    h_data = {'c': c_values}

    if func_type == 'Original Sigmoid':
        for i, theta in enumerate(thetas):
            col_name = f'h({theta},c)'
            h_data[col_name] = sigmoid(theta, c_values)
    else:
        h_data['h_smooth(c)'] = smooth_transition(c_values)

    df = pd.DataFrame(h_data).set_index('c')
    st.dataframe(df)

    # Download CSV
    csv = df.to_csv().encode('utf-8')
    st.download_button(label='Download CSV', data=csv, file_name='interpolation_function.csv', mime='text/csv')

    # Description and citation
    st.write("### Description:")
    st.latex(r'h = \frac{1}{1 + e^{-\theta (c - c_0)}}')
    st.write("This interpolation function maps the mole fraction of Sn \( c \) to a smooth value \( h \in [0,1] \).")
    st.write("The transition occurs around \( c_0 \approx 0.714 \) for Li₂Sn₅. The parameter \( \theta \) controls the steepness of the curve.")
    if func_type == 'Smooth Transition':
        st.write("In the smooth transition, \( \theta \) is automatically calculated to ensure:")
        st.latex(r'h(0.71) \approx 0.01, \quad h(0.95) \approx 0.99')

    st.write("### How to Cite This Work:")
    st.markdown("""
    If you find this work useful, please cite the following paper:

    ```
    A. Kunwar and N. Moelans, Phase field modeling of elastochemical effects at the Sn anode of Lithium-ion batteries (Working Title), 2025 (Work in Progress)
    ```
    """)

if __name__ == '__main__':
    main()
