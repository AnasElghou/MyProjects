# Physics-Informed Neural Network (PINN) for Viscous Burgers' Equation

A TensorFlow 2 implementation of a Physics-Informed Neural Network (PINN) designed to solve the 1D viscous Burgers' equation—a fundamental partial differential equation (PDE) in fluid mechanics, acoustics, and non-linear wave propagation.

---

## 📌 Problem Formulation

The 1D viscous Burgers' equation is given by:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, \quad x \in [-1, 1], \quad t \in [0, 1]$$

where:
- $u(x, t)$ is the velocity profile.
- $\nu = \frac{0.1}{\pi}$ is the kinematic viscosity coefficient.

### Initial and Boundary Conditions

* **Initial Condition (IC):**
  $$u(x, 0) = -\sin(\pi x), \quad x \in [-1, 1]$$
* **Dirichlet Boundary Conditions (BC):**
  $$u(-1, t) = u(1, t) = 0, \quad t \in [0, 1]$$

---

## 🧠 Model Architecture & Methodology

The network takes spatial coordinate $x$ and temporal coordinate $t$ as inputs and predicts the velocity $u(x, t)$.

```
Input (x, t) ──> [Dense(32, tanh)] ──> [Dense(32, tanh)] ──> [Dense(32, tanh)] ──> Output u(x, t)
```

### Physics-Informed Loss Function
The total loss function evaluated during training consists of three Mean Squared Error (MSE) terms:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{IC}} + \mathcal{L}_{\text{BC}} + \mathcal{L}_{\text{PDE}}$$

1. **Initial Condition Loss ($\mathcal{L}_{\text{IC}}$):** Measures deviation from $u(x, 0) = -\sin(\pi x)$.
2. **Boundary Condition Loss ($\mathcal{L}_{\text{BC}}$):** Enforces zero velocity at spatial boundaries $x = \pm 1$.
3. **PDE Residual Loss ($\mathcal{L}_{\text{PDE}}$):** Computed across random interior collocation points using nested `tf.GradientTape` for automatic differentiation:

$$f(x, t) = \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2}$$

$$\mathcal{L}_{\text{PDE}} = \frac{1}{N} \sum_{i=1}^N |f(x_i, t_i)|^2$$

---

## 🛠️ Prerequisites & Dependencies

Make sure you have the following Python libraries installed:

```bash
pip install tensorflow numpy matplotlib
```

---

## 🚀 Training Configuration

* **Optimizer:** Adam ($\text{learning rate} = 0.01$)
* **Epochs:** 5000
* **Sampling:**
  * Initial points: 50 randomly sampled points at $t=0$
  * Boundary points: 50 randomly sampled points at $x = -1$ and $x = 1$
  * Collocation points: 1000 uniform random points across $(x, t) \in [-1, 1] \times [0, 1]$

---

## 📊 Results & Visualization

The PINN accurately captures the steepening of the wave front into a shock wave near $x = 0$ as time progresses.

### 1. Velocity Profile Evolution (2D Slice)
Shows how the initial sinusoidal velocity field evolves and steepens over time slices $t \in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]$:

![2D Wave Profile Evolution](https://user-images.githubusercontent.com/placeholder/burgers_2d.png) <!-- Update image path if hosted -->

### 2. Spatiotemporal Solution Surface (3D Plot)
3D surface visualization displaying the continuous output $u(x, t)$ over the domain $(x, t)$:

![3D Surface Prediction](https://user-images.githubusercontent.com/placeholder/burgers_3d.png) <!-- Update image path if hosted -->

---

## 📂 Repository Structure

```text
.
├── Burgers_PINN.ipynb    # Jupyter Notebook containing full implementation and plots
└── README.md             # Project documentation
```

---

## 📄 References

* Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). **Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.** *Journal of Computational Physics*, 378, 686-707.