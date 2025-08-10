"""Simple Federated Averaging implementation with pure Python.

The script simulates several clients training a linear regression model on
synthetic data. The server aggregates client weights using the FedAvg
algorithm (weighted average by local dataset sizes).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Client:
    """Client participating in federated learning."""

    data: List[Tuple[float, float]]
    lr: float = 0.1
    epochs: int = 5

    def train(self, weight: float) -> Tuple[float, int]:
        """Train local model starting from ``weight``.

        Returns the updated weight and number of samples used.
        """
        w = weight
        n = len(self.data)
        for _ in range(self.epochs):
            grad = sum((w * x - y) * x for x, y in self.data) / n
            w -= self.lr * grad
        return w, n


class Server:
    def __init__(self, clients: List[Client], init_weight: float = 0.0):
        self.clients = clients
        self.weight = init_weight

    def round(self) -> float:
        total = 0
        weighted_sum = 0.0
        for client in self.clients:
            w, n = client.train(self.weight)
            weighted_sum += w * n
            total += n
        self.weight = weighted_sum / total
        return self.weight


def simulate(num_rounds: int = 5, num_clients: int = 3, seed: int | None = None) -> float:
    random.seed(seed)
    true_w = 2.0
    clients: List[Client] = []
    for _ in range(num_clients):
        data = []
        for _ in range(20):
            x = random.gauss(0, 1)
            noise = random.gauss(0, 0.1)
            y = true_w * x + noise
            data.append((x, y))
        clients.append(Client(data))

    server = Server(clients)
    for _ in range(num_rounds):
        server.round()
    return server.weight


if __name__ == "__main__":
    final_weight = simulate(num_rounds=10, seed=42)
    print(f"Trained global weight: {final_weight:.3f}")
