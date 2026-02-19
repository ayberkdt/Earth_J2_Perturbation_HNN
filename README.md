# OrbitPINN: Physics-Informed Neural Networks for Satellite Orbit Prediction (Two-Body + J2)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange)](https://github.com/)

**OrbitPINN**, LEO/MEO yörüngeleri için **Two-Body (Kepler) + J2 pertürbasyonu** içeren dinamikleri öğrenmeyi hedefleyen bir **Physics-Informed Neural Network (PINN)** iskeletidir.

Salt veri odaklı yaklaşımların aksine, Newton mekaniği ve **J2 (Dünya’nın basıklığı)** terimini kayıp fonksiyonuna entegre ederek, veri az olduğunda bile fiziksel olarak tutarlı yörünge tahminleri üretmeyi amaçlar.

---

## 🚀 Özellikler

- **Fizik modeli:** Two-Body + **J2** (SciPy `solve_ivp` ile ground-truth)
- **Mimari seçenekleri:** MLP / Fourier / SIREN / DeepONet / Phase features
- **Canonical units (boyutsuzlaştırma):**
  - `DU = r_ref_km`, `TU = sqrt(DU^3 / μ)`, `VU = DU/TU`
  - eğitim stabilitesi ve ölçek tutarlılığı için
- **Kayıp bileşenleri:**
  - veri kaybı `L_data`
  - fizik kaybı `L_phys` (AutoGrad türev + dinamik kıyas)
  - opsiyonel enerji regülarizasyonu `L_energy`
- **Adaptive weighting (opsiyonel):** veri/fizik terimlerini eğitim sırasında dinamik dengeleme
- **Hibrit optimizer (opsiyonel):** Adam → L-BFGS geçişi
- **PyQt6 UI:** dataset üretimi, eğitim ve değerlendirme için tek panel

---

## 📦 Proje Yapısı

```text
OrbitPINN/
  orbit_core.py          # çekirdek fizik + ölçekleme + dataset + modeller + loss + checkpoint
  generate_dataset.py    # solve_ivp ile CSV + meta JSON üretir
  train_pinn.py          # eğitim (checkpoint + loss history + meta)
  evaluate_pinn.py       # checkpoint yükle + truth solve_ivp + metrikler/plotlar
  ui.py                  # PyQt6 arayüz
  dataset/               # üretilen csv + meta
  checkpoints/           # eğitim çıktıları
  eval_outputs/          # değerlendirme çıktıları
