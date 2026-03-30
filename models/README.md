# This directory holds trained model files and metrics.
# 
# Files tracked in git:
#   stress_model_metrics.json  — training metrics (small JSON)
#
# Files NOT tracked (too large / auto-generated):
#   face_landmarker.task       — download with the command in README (~25 MB)
#   stress_model.pkl           — auto-generated on first run by _ensure_model()
#
# To regenerate the model:
#   python train.py
