"""Candidate registry — maps names to Candidate factory classes.

Only the nine baselines reported in the paper are registered:
  (A) Geometry-aware 6D pose : GDR-Net, SC6D, HccePose
  (B) 3D keypoint regression : REDE, Uni6D, FFB6D
  (C) Foundation-style pose  : MegaPose, GigaPose, FoundationPose
"""

REGISTRY: dict[str, str] = {
    # (A) Geometry-aware 6D pose
    "A1_gdrnet":   "common.candidates.a1_gdrnet:A1GdrnetCandidate",
    "A2_sc6d":     "common.candidates.a2_sc6d:A2Sc6dCandidate",
    "A3_hccepose": "common.candidates.a3_hccepose:A3HcceposeCandidate",

    # (B) 3D keypoint regression
    "B1_rede":   "common.candidates.b1_rede:B1RedeCandidate",
    "B2_uni6d":  "common.candidates.b3_uni6d:B3Uni6DCandidate",
    "B3_ffb6d":  "common.candidates.b3_ffb6d:B3Ffb6dCandidate",

    # (C) Foundation-style 6D pose
    "C1_megapose_official":       "common.candidates.c1_megapose_official:C1MegaPoseOfficialCandidate",
    "C2_gigapose_official":       "common.candidates.c2_gigapose_official:C2GigaPoseOfficialCandidate",
    "C3_foundationpose_official": "common.candidates.c3_foundationpose_official:C3FoundationPoseOfficialCandidate",
}


def load_candidate(name: str):
    import importlib

    if name not in REGISTRY:
        raise KeyError(f"unknown candidate {name!r}. registered: {list(REGISTRY)}")
    spec = REGISTRY[name]
    mod_name, cls_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)()
