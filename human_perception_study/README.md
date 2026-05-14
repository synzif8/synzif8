# Human Perception Study

This directory contains the **analysis notebooks** for the N=6 domain-expert
study reported in the paper (Real-vs-Synthetic classification and edge-length
estimation).

| Notebook | Purpose |
|---|---|
| `human_exp_Real_Fake.ipynb` | Real-vs-Synthetic classification responses |
| `human_exp_Edge_esti.ipynb` | Edge-length estimation responses |
| `human_perception_analysis.ipynb` | Aggregated tables / figures for the paper |

## Data-collection interface

The GUI used to collect the participant responses is the FastAPI web tool in
`../annotation_tool/server.py` (vertex annotation) and
`../annotation_tool/server_length.py` (edge-length annotation). The same
codebase served both the per-instance annotation and the timed perception
study sessions.

To launch the annotation server locally:

```bash
cd <PROJECT_ROOT>/annotation_tool
uvicorn server:app --port 8000
# open http://localhost:8000 in a browser
```
