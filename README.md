# MIoT-IDS: Federated, Drift-Aware, Explainable Intrusion Detection for Medical IoT

## Why this is different from typical IoT-IDS research

| Typical published IDS | This project |
|---|---|
| Train + test split on the *same* static dataset | Train federated across simulated "hospitals/devices", test on held-out **unseen** device traffic |
| All data pooled centrally | Each client trains locally; only model weights are shared (FedAvg) |
| Model trained once, never updated | Streaming drift detector (ADWIN) triggers local retraining when traffic shifts |
| Black-box "ATTACK/NORMAL" output | Every alert ships with a SHAP-based reason ("packet_iat 4.2σ above device baseline") |
| Heavy model, cloud-only | Small model (few KB after quantization) — runs on constrained devices |

## Architecture

```
 Device A (ICU monitor)      Device B (Infusion pump)      Device C (BLE sensor)
       |  local model              |  local model               |  local model
       |  (trained on local        |  (trained on local          |  (trained on local
       |   traffic only)           |   traffic only)             |   traffic only)
       v                           v                              v
   ------------------------- FedAvg aggregation server -------------------------
                          (only weights exchanged, never raw traffic)
                                       |
                                       v
                         Global model redistributed to clients
                                       |
                                       v
                     Each client also runs a local ADWIN drift
                     detector -> flags "retrain me" when traffic
                     distribution changes (new attack pattern,
                     firmware update, etc.)
                                       |
                                       v
                   On alert: SHAP explains which feature(s)
                   drove the decision -> shown to the admin
```

## Project layout

```
miot-ids/
├── README.md
├── requirements.txt
├── config.yaml                 # central config: rounds, clients, model size, thresholds
├── data/
│   └── README.md                # where/how to get real datasets (Edge-IIoTset, TON_IoT, MedBIoT)
├── src/
│   ├── data_loader.py           # loads + partitions traffic data per simulated client
│   ├── features.py              # protocol-aware feature extraction (MQTT/CoAP-friendly)
│   ├── model.py                 # lightweight autoencoder / classifier (PyTorch)
│   ├── federated.py             # FedAvg client + server simulation
│   ├── drift.py                 # ADWIN-based concept drift detector
│   ├── explain.py               # SHAP wrapper -> human-readable alert reasons
│   └── train.py                 # entry point: ties it all together
└── models/                      # saved global model checkpoints
```

## How to run (once you plug in a real dataset)

```bash
pip install -r requirements.txt
python src/train.py --config config.yaml
```

## Recommended datasets to start with

- **Edge-IIoTset** — realistic IoT/IIoT traffic, already has a federated-learning-friendly split.
- **MedBIoT** — botnet traffic specifically from medical/IoT-style devices, good for the "medical" framing.
- **TON_IoT** — good for testing cross-dataset generalization (train on Edge-IIoTset, test on TON_IoT).

Put raw CSVs in `data/raw/<dataset_name>/` — `data_loader.py` has stub loaders for each.

## What to report in your writeup to prove it's "better"

1. **Cross-dataset accuracy** (train on A, test on B) — not just same-dataset split.
2. **Federated vs centralized accuracy gap** — show federated learning loses little accuracy while preserving privacy.
3. **Drift detection latency** — how many samples after a distribution shift before ADWIN flags it and the model recovers.
4. **Explanation quality** — a few example alerts with SHAP reasons, sanity-checked by hand.
5. **Model size / inference time** on a simulated constrained device (e.g., Raspberry Pi Zero specs).
