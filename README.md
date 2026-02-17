# OrbitPINN: Physics-Informed Neural Networks for Satellite Orbit Prediction (J2 Perturbation)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange)](https://github.com/)

**OrbitPINN**, MEO (Medium Earth Orbit) ve LEO uydularının yörünge mekaniğini öğrenmek ve tahmin etmek için geliştirilmiş bir **Fizik Bilgili Sinir Ağı (Physics-Informed Neural Network - PINN)** iskeletidir. 

Bu proje, salt veri odaklı yaklaşımların aksine, Newton'un hareket yasalarını ve **J2 (Dünya'nın basıklığı)** etkisini kayıp fonksiyonuna (loss function) entegre ederek, verinin az olduğu durumlarda bile fiziksel olarak tutarlı yörünge tahminleri üretmeyi hedefler.

---

## 🚀 Özellikler & PIML Stratejileri

Bu proje standart bir MLP (Multi-Layer Perceptron) uygulamasının ötesine geçerek modern PIML tekniklerini içerir:

* **Fizik Motoru:** 2-Body (Kepler) + $J_2$ Perturbasyonu (Dünya'nın ekvatoral şişkinliği).
* **Mimari:** Yüksek frekanslı yörünge dinamiklerini yakalamak için **Fourier Feature Mapping** ve **SIREN** (Sinusoidal Representation Networks) desteği.
* **Boyutsuzlaştırma (Canonical Units):** Gradyan patlamalarını önlemek için $TU$ (Time Unit) ve $DU$ (Distance Unit) tabanlı ölçekleme.
* **Adaptive Loss Weighting:** Veri kaybı ($L_{data}$) ve Fizik kaybı ($L_{phys}$) arasındaki dengeyi eğitim sırasında dinamik olarak ayarlayan mekanizma.
* **Symplectic Korunum (Opsiyonel):** Sistemin toplam enerjisinin (Hamiltonian) korunmasına dair ek regülarizasyon ($L_{energy}$).
* **Hibrit Optimizer:** Global arama için `Adam`, hassas yakınsama için `L-BFGS` (Second-order optimizer).

---

## 📚 Matematiksel Altyapı

Model, aşağıdaki diferansiyel denklemi (ODE) çözmeyi öğrenir:

$$
\ddot{\mathbf{r}} = -\frac{\mu}{r^3}\mathbf{r} + \mathbf{a}_{J2}(\mathbf{r})
$$

Burada $J_2$ ivmesi (Kartezyen formda):

$$
\mathbf{a}_{J2} = \frac{3}{2} J_2 \left(\frac{\mu}{r^2}\right) \left(\frac{R_E}{r}\right)^2 
\begin{bmatrix} 
\frac{x}{r}(5\frac{z^2}{r^2} - 1) \\ 
\frac{y}{r}(5\frac{z^2}{r^2} - 1) \\ 
\frac{z}{r}(5\frac{z^2}{r^2} - 3) 
\end{bmatrix}
$$

Model, $t$ zaman girdisine karşılık durum vektörünü $\mathbf{s} = [x, y, z, \dot{x}, \dot{y}, \dot{z}]^T$ tahmin eder ve otomatik türev (AutoGrad) ile hesaplanan ivmeyi yukarıdaki fiziksel modelle kıyaslar.

---

## 🛠️ Kurulum

Projeyi klonlayın ve bağımlılıkları yükleyin:

```bash
git clone [https://github.com/kullaniciadi/OrbitPINN.git](https://github.com/kullaniciadi/OrbitPINN.git)
cd OrbitPINN
pip install -r requirements.txt
