# PathSnap features

`features.py` is the reusable feature extractor. `features_notebook.ipynb` runs and audits it.

Feature table grain: one GPS observation x one candidate road segment.

Features include GPS timing/motion, distance to candidate arc and endpoints, projection along arc, arc length/bearing, heading error and related ratios.

The `.route` file is not treated as row-aligned with `.track`; supervised `reference_arc_id` creation is a separate documented step.

Use the same feature table for the requested model comparison. Scale only for scale-sensitive models, fitting preprocessing on permitted training trips only.
