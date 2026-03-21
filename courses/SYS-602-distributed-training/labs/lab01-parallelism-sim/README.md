# Lab 01 — Parallelism simulation

| Script | Purpose |
|--------|---------|
| `dp_simulation.py` | Data-parallel AllReduce volume |
| `tp_simulation.py` | TP AllReduce per layer (toy) |
| `pp_simulation.py` | Pipeline bubble fraction |
| `memory_calculator.py` | ZeRO 0–3 per-GPU memory |

Run examples:

```bash
python dp_simulation.py --world-size 64 --params-gb 8
python memory_calculator.py --stage 3 --world-size 512
```
