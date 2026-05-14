"""
Rhombic Vertex Annotation Tool for SEM Images


Workflow:
  1. Load SEM images from SEM_IMAGE_DIR
  2. Run instance segmentation (Mask R-CNN) to detect rhombics
  3. User selects well-segmented rhombics
  4. User manually clicks 2D vertices on each selected rhombic
  5. Save annotations (rhombic ID + mask + vertex coords) to JSON

★ Segmentation Model Connection Point ★
  - Set SEG_CHECKPOINT env var or modify the default path below
  - Currently: seg_exp05 Mask R-CNN 8k data (ResNet-50 + FPN v2)
  - When your new model is ready, just change the checkpoint path

Usage:
  cd <PROJECT_ROOT>/annotation_tool
  python vertex_annotator/server.py
"""

import os, sys, io, json, glob, base64
import numpy as np
import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

# ============================================================================
# Configuration
# ============================================================================

SEM_IMAGE_DIR = os.environ.get('SEM_IMAGE_DIR', 'sem_images')
ANNOTATION_DIR = os.environ.get('ANNOTATION_DIR', 'vertex_annotations')
SEG_THRESHOLD = float(os.environ.get('SEG_THRESHOLD', '0.5'))
MIN_MASK_AREA = int(os.environ.get('MIN_MASK_AREA', '50'))

# ──────────────────────────────────────────────────────────────
# ★ SEGMENTATION MODEL — Change this path when new model ready
# ──────────────────────────────────────────────────────────────
SEG_CHECKPOINT = os.environ.get('SEG_CHECKPOINT',
    '<PROJECT_ROOT>/segmentation/models/best_model.pth')

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')

# ============================================================================
# Setup
# ============================================================================

os.makedirs(ANNOTATION_DIR, exist_ok=True)
os.makedirs(SEM_IMAGE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# Load segmentation model
# ============================================================================

seg_model = None
if os.path.exists(SEG_CHECKPOINT):
    seg_model = maskrcnn_resnet50_fpn_v2(weights=None)
    in_f = seg_model.roi_heads.box_predictor.cls_score.in_features
    seg_model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    in_fm = seg_model.roi_heads.mask_predictor.conv5_mask.in_channels
    seg_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_fm, 256, 2)
    ckpt = torch.load(SEG_CHECKPOINT, map_location=device, weights_only=False)
    seg_model.load_state_dict(ckpt['model_state_dict'])
    seg_model.to(device).eval()
    print(f"[SEG] Loaded: {SEG_CHECKPOINT} (epoch {ckpt['epoch']})")
else:
    print(f"[SEG] WARNING: Checkpoint not found at {SEG_CHECKPOINT}")
    print(f"[SEG] Segmentation unavailable — place checkpoint and restart")

# ============================================================================
# Helpers
# ============================================================================

def np_to_b64(img, fmt='png'):
    pil = Image.fromarray(img, 'L' if img.ndim == 2 else 'RGB')
    buf = io.BytesIO()
    pil.save(buf, format=fmt.upper())
    return base64.b64encode(buf.getvalue()).decode()


def get_image_list():
    exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(SEM_IMAGE_DIR, ext)))
        files.extend(glob.glob(os.path.join(SEM_IMAGE_DIR, ext.upper())))
    return sorted(set(os.path.basename(f) for f in files))


def annotation_path(image_name):
    return os.path.join(ANNOTATION_DIR, Path(image_name).stem + '.json')


def load_annotation(image_name):
    p = annotation_path(image_name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

# ============================================================================
# FastAPI
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Rhombic Vertex Annotator")
app.mount("/static", StaticFiles(directory=os.path.join(SCRIPT_DIR, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(SCRIPT_DIR, "static", "index.html")) as f:
        return f.read()


@app.get("/api/images")
async def list_images():
    names = get_image_list()
    result = []
    for name in names:
        ann = load_annotation(name)
        n_done = 0
        if ann:
            n_done = sum(1 for r in ann.get('rhombics', []) if r.get('vertices'))
        result.append({'name': name, 'annotated': ann is not None, 'n_vertices_done': n_done})
    return JSONResponse({'images': result, 'total': len(result),
                         'sem_dir': os.path.abspath(SEM_IMAGE_DIR)})


@app.post("/api/segment")
async def segment(image_name: str = Form(...)):
    img_path = os.path.join(SEM_IMAGE_DIR, image_name)
    if not os.path.exists(img_path):
        return JSONResponse({'error': f'Image not found: {image_name}'}, status_code=404)
    if seg_model is None:
        return JSONResponse({'error': 'Segmentation model not loaded. '
                             'Set SEG_CHECKPOINT and restart.'}, status_code=503)

    img = np.array(Image.open(img_path).convert('L'))
    H, W = img.shape

    rn = img.astype(np.float32) / 255.0
    t = torch.from_numpy(rn).unsqueeze(0).repeat(3, 1, 1).float().to(device)
    with torch.no_grad():
        pred = seg_model([t])[0]

    sc = pred['scores'].cpu().numpy()
    objects = []
    for i in range(len(sc)):
        if sc[i] > SEG_THRESHOLD:
            m = (pred['masks'][i, 0].cpu().numpy() > 0.5).astype(np.uint8)
            if m.sum() > MIN_MASK_AREA:
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polys = []
                for c in cnts:
                    eps = 0.002 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, eps, True)
                    pts = approx.squeeze().tolist()
                    if isinstance(pts[0], (int, float)):
                        pts = [pts]
                    polys.append(pts)
                ys, xs = np.nonzero(m)
                objects.append({
                    'idx': len(objects),
                    'score': float(sc[i]),
                    'bbox': pred['boxes'][i].cpu().numpy().tolist(),
                    'cx': float(xs.mean()), 'cy': float(ys.mean()),
                    'area': int(m.sum()),
                    'contours': polys,
                })

    return JSONResponse({
        'image': np_to_b64(img),
        'width': W, 'height': H,
        'n_objects': len(objects),
        'objects': objects,
        'existing_annotation': load_annotation(image_name),
    })


@app.post("/api/save")
async def save(data: str = Form(...)):
    ann = json.loads(data)
    name = ann.get('image_name')
    if not name:
        return JSONResponse({'error': 'image_name required'}, status_code=400)
    p = annotation_path(name)
    with open(p, 'w') as f:
        json.dump(ann, f, indent=2, ensure_ascii=False)
    return JSONResponse({'status': 'saved', 'path': os.path.abspath(p)})


@app.get("/api/export_all")
async def export_all():
    """Export all annotations as a single JSON for downstream use."""
    names = get_image_list()
    all_data = []
    for name in names:
        ann = load_annotation(name)
        if ann and ann.get('rhombics'):
            all_data.append(ann)
    return JSONResponse({'total_images': len(all_data),
                         'annotations': all_data})


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7968)
