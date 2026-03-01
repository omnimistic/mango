# 🥭 Project Mango: Local Text-to-Image Inference Engine

Project Mango is a high-performance, CPU-optimized AI art generation suite. Unlike standard implementations that rely on high-level library wrappers, Mango features a **raw PyTorch inference pipeline** and a custom **Server-Sent Events (SSE) streaming architecture** to provide real-time feedback during the denoising process.

## 🛠️ Technical Architecture

* **Custom Inference Loop:** Built using discrete components (UNet, VAE, CLIP) to bypass standard black-box pipelines, allowing for manual tensor manipulation.
* **Dynamic CFG Routing:** Implements automated routing logic that dynamically adjusts the Classifier-Free Guidance (CFG) scale per request to vary output stylization (Dreamy, Standard, or Crunchy).
* **Real-Time SSE Streaming:** Uses Server-Sent Events to pipe mathematical progress from the Python backend directly to the frontend, enabling a live-updating progress bar.
* **Decoupled Frontend:** A modern, responsive interface built with a premium glassmorphism aesthetic, served via an optimized Flask routing system.

## 📂 Project Structure

```text
mango/
├── frontend/
│   ├── static/          # CSS and JavaScript
│   │   ├── style.css
│   │   └── script.js
│   └── index.html       # Primary Application UI
├── output/              # Local directory for generated media
├── backend.py           # Flask server and SSE routing logic
├── mango.py             # Core PyTorch Diffusion Engine
└── README.md            # Project documentation

```

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* RAM: 8GB minimum (16GB recommended)
* Storage: ~2GB for model weights

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/omnimistic/mango.git
cd mango

```


2. **Install dependencies:**
```bash
pip install torch diffusers transformers flask pillow

```


3. **Launch the Engine:**
```bash
python backend.py

```


4. **Access the UI:**
Open `http://127.0.0.1:5000` in your web browser.

## 🌐 Network Access

By default, the server is locked to `127.0.0.1` (localhost) for security. To access the engine from other devices on your local network:

1. In `backend.py`, change `host='127.0.0.1'` to `host='0.0.0.0'`.
2. Navigate to your machine's local IP (eg: `http://192.168.1.XX:5000`) on your mobile device or tablet to access the interface.

## ⚖️ License & Acknowledgements

* **The Code:** Licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
* **The Model:** This project utilizes `bk-sdm-tiny`, a derivative of Stable Diffusion. Use of these weights is subject to the **CreativeML Open RAIL-M** license. Users are responsible for ensuring generated content complies with the ethical use-case restrictions outlined therein.
