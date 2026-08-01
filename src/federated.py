"""
Minimal FedAvg implementation -- no external federated-learning framework
required, so it's easy to read and modify. Swap in Flower/PySyft later if you
want real multi-machine deployment; this version simulates all clients in one
process, which is enough to prove the concept and measure the
federated-vs-centralized accuracy gap.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import LightIDSAutoencoder


def train_local(model, X_train, epochs, lr, batch_size):
    """Train one client's model on its own local (normal-only) traffic."""
    model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def fedavg(state_dicts):
    """Average model weights across clients -- the core of FedAvg."""
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state:
        avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


def run_federated_training(client_data, config):
    global_model = LightIDSAutoencoder(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
    )

    print(f"Model size: {global_model.num_params()} parameters "
          f"(~{global_model.num_params() * 4 / 1024:.1f} KB in fp32)")

    for round_num in range(config["federated_rounds"]):
        local_states = []
        for client in client_data:
            # Autoencoder trains on NORMAL traffic only -- filter y==0 if labels exist
            X_normal = client["X_train"]
            local_state = train_local(
                global_model, X_normal,
                epochs=config["local_epochs"],
                lr=config["client_lr"],
                batch_size=config["batch_size"],
            )
            local_states.append(local_state)

        global_state = fedavg(local_states)
        global_model.load_state_dict(global_state)

        if round_num % 5 == 0 or round_num == config["federated_rounds"] - 1:
            print(f"  Round {round_num + 1}/{config['federated_rounds']} complete")

    return global_model
