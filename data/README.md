# Datasets

This project doesn't ship raw traffic data (too large, and licensing varies).
Download one of these and place it as shown:

## Edge-IIoTset (recommended starting point)
- Realistic IoT/IIoT traffic with normal + 14 attack types, already has a
  federated-learning-friendly structure.
- Search: "Edge-IIoTset dataset" (IEEE DataPort / Kaggle mirrors exist).
- Place CSV at: `data/raw/edge_iiotset/edge_iiotset.csv`

## MedBIoT (for the medical-device framing specifically)
- Botnet traffic captured from medical/IoT-style device emulations.
- Place at: `data/raw/medbiot/`

## TON_IoT (use as a cross-dataset generalization test)
- Train your model on Edge-IIoTset, then evaluate on TON_IoT to see if it
  actually generalizes rather than just memorizing one dataset's quirks.
- Place at: `data/raw/ton_iot/`

## If you have real device traffic instead
Even better -- capture actual MQTT/CoAP/BLE traffic from real or simulated
medical IoT devices using something like `tshark`/`tcpdump`, then extract
protocol-specific features (message type, payload size, timing) instead of
generic NetFlow features. Add a loader for it in `src/data_loader.py` following
the same pattern as the existing `load_*` functions.
