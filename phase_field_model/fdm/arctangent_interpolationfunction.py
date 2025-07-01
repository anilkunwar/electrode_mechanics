import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Arctangent-based interpolation with theta return
def arctan_transition(c, c1=0.71, c2=0.95, h_target=0.99):
    c0 = (c1 + c2) / 2
    arctan_arg = np.tan(np.pi * (h_target - 0.5))
    theta = arctan_arg / (c2 - c0)
    h = (1 / np.pi) * np.arctan(theta * (c - c0)) + 0.5
    return h, theta

def main():
    st.title('Arctangent Interpolation Function between Phases')
    st.write("This app uses an arctangent-based function to create a smooth interpolation between Sn and Li₂Sn₅ phases.")

    # Sidebar controls
    xf = st.sidebar.slider('x-text position', 0.0, 1.0, 0.77)
    yf = st.sidebar.slider('y-text position', 0.0, 1.0, 0.4)
    curve_color = st.sidebar.color_picker("Pick line color", "#008B8B")

    # Generate curve
    c_values = np.linspace(0, 1, 400)
    h_values, theta = arctan_transition(c_values)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(c_values, h_values, color=curve_color, linewidth=4, 
            label=fr"$\theta$ = {theta:.2f}")

    ax.set_xlabel('$c$', fontsize=25)
    ax.set_ylabel('$h$', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20, width=5.0, size=8)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle='--', linewidth=1.0)

    for spine in ax.spines.values():
        spine.set_linewidth(4)

    ax.text(xf, yf, r'$h = \frac{1}{\pi} \arctan\left( \theta (c - c_0) \right) + \frac{1}{2}$',
            fontsize=20, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')

    st.pyplot(fig)

    # Display values
    st.write("### Table of \( h(c) \) for selected \( c \) values:")
    sample_c = np.round(np.linspace(0, 1, 21), 3)
    h_sample, _ = arctan_transition(sample_c)
    df = pd.DataFrame({'c': sample_c, 'h(c)': np.round(h_sample, 5)}).set_index('c')
    st.dataframe(df)

    csv = df.to_csv().encode('utf-8')
    st.download_button(label="Download CSV", data=csv, file_name="arctan_interpolation.csv", mime="text/csv")

    # Description
    st.write("### Description:")
    st.latex(r"h(c) = \frac{1}{\pi} \arctan\left( \theta (c - c_0) \right) + \frac{1}{2}")
    st.write("Where:")
    st.latex(r"c_0 = \frac{c_1 + c_2}{2}, \quad \text{with } c_1 = 0.71, \, c_2 = 0.95")
    st.write("θ is auto-calculated so that:")
    st.latex(r"h(0.71) \approx 0.01, \quad h(0.95) \approx 0.99")
    st.write(f"Automatically computed value: $\\theta$ = **{theta:.4f}**")

    # Citation
    st.write("### How to Cite:")
    st.markdown("""
    If you use this in your work, please cite:

    ```
    A. Kunwar and N. Moelans, Phase field modeling of elastochemical effects at the Sn anode of Lithium-ion batteries (Working Title), 2025 (Work in Progress)
    ```
    """)

if __name__ == '__main__':
    main()
