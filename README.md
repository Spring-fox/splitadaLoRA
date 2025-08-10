# splitadaLoRA

Simple experiments related to SplitAdaLoRA.

## Federated Averaging Example

This repository now includes a minimal implementation of the FedAvg algorithm
in `fedavg.py`. The script simulates several clients that train local linear
regression models and then aggregates them on a server using weighted
averaging.

Run the simulation with:

```bash
python fedavg.py
```
