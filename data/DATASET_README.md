# PEMS04 Dataset

The project uses the PEMS04 files supplied with the PBL dataset archive. The executable loader validates the expected shape at runtime.

Expected files:

- `data/raw/PEMS04/PEMS04.npz`
- `data/raw/PEMS04/PEMS04.csv`

Expected tensor shape: `[16992, 307, 3]`.

Expected graph edge count: 340.

The three variables are treated as flow, speed and occupancy according to the standard PEMS04 representation used by the traffic-forecasting literature.
