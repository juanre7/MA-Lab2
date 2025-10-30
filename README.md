# Lab 2 – 2D Convolution and FIR Filters

This repository contains a single MATLAB/Octave script that demonstrates core image filtering concepts and produces five figures for your report. No toolboxes are required. The script also includes minimal helper functions so it runs in base MATLAB or Octave. 

## How to run

1. Rename the provided file to `Lab2.m` or open it directly in MATLAB and run it.
2. The script will load `peppers.png` or `cameraman.tif` if available. If neither is found, it generates a synthetic image, so the demo always runs. 
3. Five figures will open, corresponding to sections 1, 2, 3, 4, and 6 in the script. Section 5 prints a numerical check to the console but does not create a figure. 

> Tip: To save each figure, use `File > Save As...` or programmatically after each figure is created with `exportgraphics(gcf,'figX_name.png','Resolution',300)`. 

---


### Figure 1: Impulse response of a 3x3 average

Shows the impulse response obtained by convolving a delta image with a 3x3 box kernel. Useful to visualize the filter’s point spread. 

<img width="764" height="693" alt="Figure_1" src="https://github.com/user-attachments/assets/7689f434-5f08-4547-9338-a156ec2e5a2c" />


---

### Figure 2: Low-pass filtering – box vs Gaussian

Four panels: Original, 3x3 box, 7x7 box, and separable Gaussian. Compares smoothing strength and visual artifacts. 

<img width="1045" height="182" alt="Figure_2" src="https://github.com/user-attachments/assets/6469c00a-8ef2-4407-9068-568e9e4960c1" />

---

### Figure 3: Unsharp masking (sharpening)

Four panels: Original, blurred image, high-frequency mask, and sharpened result. Demonstrates contrast enhancement via unsharp masking. 

<img width="1045" height="182" alt="Figure_3" src="https://github.com/user-attachments/assets/7b1552b9-ff41-43d0-9ecb-c456583917cd" />


---

### Figure 4: Edge detection – Sobel and Laplacian

Four panels: Sobel Gx, Sobel Gy, gradient magnitude, and Laplacian response. Illustrates directional gradients vs second-derivative edge cues. 

<img width="1045" height="182" alt="Figure_4" src="https://github.com/user-attachments/assets/caf41eb3-dd55-4f31-be1f-f4edb7719466" />


---

### Figure 5: Boundary handling modes

Three panels: replicate, symmetric, and circular padding. Shows how edge treatment affects filtered results near borders. 

<img width="1045" height="242" alt="Figure_5" src="https://github.com/user-attachments/assets/dab9b608-ed8a-4df4-9263-c473e749cc3c" />


---

