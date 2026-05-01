// SAM3 Annotator frontend
// Vanilla JS + Konva (CDN). Single-file app for MVP.

const CAT_COLORS = {
  Tool: '#00b4d8',
  Liver: '#e63946',
  Gallbladder: '#52b788',
  Meat: '#f77f00',
  Skin: '#ffb4a2',
  FBF: '#7209b7',
  PCH: '#ffd60a',
};
const DEFAULT_COLOR = '#ffff00';
const catColor = (c) => CAT_COLORS[c] || DEFAULT_COLOR;

// ---------- State ----------
const S = {
  episode: null,
  snippet: null,
  nFrames: 0,
  width: 0,
  height: 0,
  splitSize: null,        // null for cluster-format snippets
  framesDirName: 'frames_left',
  startFrame: 0,
  endFrame: 0,
  lightweight: false,
  categories: [],
  activeCategory: null,
  activeTool: 'brush',
  brushSize: 12,
  pointLabel: 1,
  frameIdx: 0,
  showGt: true,
  showApproved: true,
  showPropagated: true,
  // current frame data
  gt: {},          // cat -> [polygon_flat]
  approved: {},    // cat -> RLE
  propagated: {},  // cat -> RLE
  // edit-layer is in Konva — read via exportToPNG when committing
  // point prompts collected for SAM-click mode
  samPoints: [],   // [{x, y, label}]
  samBox: null,    // [x1,y1,x2,y2]
  samBoxFirstClick: null,
  // text-prompt detections (latest run)
  textDets: [],    // [{idx, score, rle, box, pixels}]
  // Categories whose GT is currently loaded onto the edit layer (so we hide
  // the dashed GT outline for those — otherwise it looks un-erasable).
  editingGT: new Set(),
  // undo stack: array of {type, payload}
  undoStack: [],
};

// ---------- DOM ----------
const $ = (sel) => document.querySelector(sel);

// ---------- Konva setup ----------
let stage, imageLayer, gtLayer, approvedLayer, propagatedLayer, editLayer, promptLayer;
let displayScale = 1;  // canvas display size / original size

function _calcStageSize(width, height) {
  const wrap = $('#canvas-wrap');
  // Use a small inset so we don't blow past the wrap padding
  const maxW = Math.max(120, wrap.clientWidth - 4);
  const maxH = Math.max(120, wrap.clientHeight - 4);
  const scale = Math.min(maxW / width, maxH / height, 1);
  return {
    scale,
    w: Math.round(width * scale),
    h: Math.round(height * scale),
  };
}

function initKonva(width, height) {
  const { scale, w, h } = _calcStageSize(width, height);
  displayScale = scale;

  $('#konva-container').innerHTML = '';
  stage = new Konva.Stage({
    container: 'konva-container',
    width: w,
    height: h,
  });
  imageLayer = new Konva.Layer({ listening: false });
  gtLayer = new Konva.Layer({ listening: false });
  approvedLayer = new Konva.Layer({ listening: false });
  propagatedLayer = new Konva.Layer({ listening: false });
  editLayer = new Konva.Layer();
  promptLayer = new Konva.Layer({ listening: false });

  stage.add(imageLayer, propagatedLayer, approvedLayer, gtLayer, editLayer, promptLayer);

  stage.on('mousedown touchstart', onPointerDown);
  stage.on('mousemove touchmove', onPointerMove);
  stage.on('mouseup touchend', onPointerUp);
}

// Re-fit the stage when viewport changes
let _resizeTimer = null;
function handleResize() {
  if (!stage || !S.snippet) return;
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(() => {
    const { scale, w, h } = _calcStageSize(S.width, S.height);
    if (Math.abs(scale - displayScale) < 0.001 && w === stage.width() && h === stage.height()) return;
    displayScale = scale;
    stage.size({ width: w, height: h });
    // Resize all layer Konva.Image children to fill new stage
    [imageLayer, gtLayer, approvedLayer, propagatedLayer, editLayer, promptLayer].forEach(layer => {
      layer.getChildren().forEach(node => {
        if (node.className === 'Image') {
          node.width(w);
          node.height(h);
        }
      });
      layer.batchDraw();
    });
    // Re-render overlays so polygons etc. use the new displayScale
    renderOverlays();
  }, 80);
}

function clearLayers() {
  if (!stage) return;
  [imageLayer, gtLayer, approvedLayer, propagatedLayer, editLayer, promptLayer].forEach(l => {
    l.destroyChildren();
    l.batchDraw();
  });
}

// ---------- API ----------
async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return r.json();
}
async function apiPost(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return r.json();
}

// ---------- Frame loading ----------
async function loadFrame(idx) {
  if (!S.snippet || idx < 0 || idx >= S.nFrames) return;
  S.frameIdx = idx;
  $('#frame-slider').value = idx;

  // Image
  const img = new Image();
  img.src = `/api/frame/${idx}?t=${Date.now()}`;
  img.onload = () => {
    imageLayer.destroyChildren();
    const k = new Konva.Image({
      image: img,
      width: stage.width(),
      height: stage.height(),
      listening: false,
    });
    imageLayer.add(k);
    imageLayer.batchDraw();
    updateFrameInfo();
  };

  // Clear edit layer on navigate
  editLayer.destroyChildren();
  promptLayer.destroyChildren();
  S.samPoints = [];
  S.samBox = null;
  S.samBoxFirstClick = null;
  S.editingGT.clear();
  editLayer.batchDraw();
  promptLayer.batchDraw();
  S.undoStack = [];

  // Load GT + masks in parallel
  const [gtResp, masksResp] = await Promise.all([
    apiGet(`/api/gt/${idx}`).catch(() => ({ polygons: {} })),
    apiGet(`/api/masks/${idx}`).catch(() => ({ approved: {}, propagated: {} })),
  ]);
  S.gt = gtResp.polygons || {};
  S.approved = masksResp.approved || {};
  S.propagated = masksResp.propagated || {};
  renderOverlays();
  // Re-render masks panel so per-frame view updates and counter reflects current frame
  renderMasksList();
}

function updateFrameInfo() {
  if (!S.snippet) return;
  const frameNum = S.startFrame + S.frameIdx;
  if (S.splitSize && S.splitSize > 0) {
    const split = Math.floor(frameNum / S.splitSize);
    const offset = frameNum % S.splitSize;
    const isKf = offset === 0 ? ' [KF]' : '';
    $('#frame-info').textContent =
      `Frame ${S.frameIdx + 1}/${S.nFrames} · #${frameNum} · split ${split} offset ${offset}${isKf}`;
  } else {
    // Cluster-format snippet: no split layout, just show frame index + abs num.
    $('#frame-info').textContent =
      `Frame ${S.frameIdx + 1}/${S.nFrames} · #${frameNum}`;
  }
}

// ---------- Overlays ----------
function renderOverlays() {
  gtLayer.destroyChildren();
  approvedLayer.destroyChildren();
  propagatedLayer.destroyChildren();

  if (S.showGt) {
    for (const [cat, polys] of Object.entries(S.gt)) {
      // If the user has pulled this GT onto the edit layer, skip drawing the
      // outline so the edit-layer raster is the only render of this mask.
      if (S.editingGT.has(cat)) continue;
      const col = catColor(cat);
      for (const flat of polys) {
        const pts = flat.map(v => v * displayScale);
        gtLayer.add(new Konva.Line({
          points: pts, stroke: col, strokeWidth: 2,
          closed: true, fill: col + '22', listening: false,
        }));
      }
    }
  }
  if (S.showApproved) {
    for (const [cat, rle] of Object.entries(S.approved)) {
      addRleToLayer(approvedLayer, rle, catColor(cat), 0.55);
    }
  }
  if (S.showPropagated) {
    for (const [cat, rle] of Object.entries(S.propagated)) {
      addRleToLayer(propagatedLayer, rle, catColor(cat), 0.30);
    }
  }
  gtLayer.batchDraw();
  approvedLayer.batchDraw();
  propagatedLayer.batchDraw();
}

// COCO RLE client-side decoder (column-major) → returns ImageData
function decodeRLE(rle) {
  const [h, w] = rle.size;
  const counts = rle.counts;
  // decode the string into a Uint8Array using the COCO ASCII scheme
  // Ref: https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c
  const cnts = [];
  let i = 0;
  while (i < counts.length) {
    let x = 0, k = 0, more = 1;
    while (more) {
      const c = counts.charCodeAt(i) - 48;
      x |= (c & 0x1f) << (5 * k);
      more = c & 0x20;
      i++;
      k++;
      if (!more && (c & 0x10)) x |= -1 << (5 * k);
    }
    if (cnts.length > 2) x += cnts[cnts.length - 2];
    cnts.push(x);
  }
  // cnts alternates: 0s-count, 1s-count, ...
  const mask = new Uint8Array(w * h);
  let p = 0, v = 0;
  for (const c of cnts) {
    for (let j = 0; j < c; j++) mask[p++] = v;
    v = 1 - v;
  }
  // COCO is column-major; convert to row-major RGBA ImageData
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const col = mask[x * h + y];
      if (col) {
        const di = (y * w + x) * 4;
        data[di] = 255; data[di+1] = 255; data[di+2] = 255; data[di+3] = 255;
      }
    }
  }
  return new ImageData(data, w, h);
}

function addRleToLayer(layer, rle, color, opacity) {
  try {
    const imgData = decodeRLE(rle);
    // color the white pixels
    const col = hexToRgb(color);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] > 0) {
        d[i] = col.r; d[i + 1] = col.g; d[i + 2] = col.b;
        d[i + 3] = Math.round(opacity * 255);
      }
    }
    const cvs = document.createElement('canvas');
    cvs.width = imgData.width;
    cvs.height = imgData.height;
    cvs.getContext('2d').putImageData(imgData, 0, 0);
    layer.add(new Konva.Image({
      image: cvs,
      width: stage.width(),
      height: stage.height(),
      listening: false,
    }));
  } catch (e) {
    console.error('RLE render failed', e);
  }
}
function hexToRgb(hex) {
  const m = /#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(hex);
  return m ? { r: parseInt(m[1],16), g: parseInt(m[2],16), b: parseInt(m[3],16) } : {r:255,g:255,b:0};
}

// ---------- Drawing state machine ----------
let painting = false;
let lastPos = null;
let currentStroke = null;

function stageToOriginal(pos) {
  return { x: pos.x / displayScale, y: pos.y / displayScale };
}

function onPointerDown(e) {
  if (!S.snippet) return;
  const pos = stage.getPointerPosition();
  if (!pos) return;

  if (S.activeTool === 'brush' || S.activeTool === 'eraser') {
    // Snapshot pre-stroke for unified undo
    pushEditLayerUndo(S.activeTool);
    painting = true;
    lastPos = pos;
    currentStroke = new Konva.Line({
      points: [pos.x, pos.y],
      stroke: S.activeTool === 'eraser' ? 'black' : catColor(S.activeCategory),
      strokeWidth: S.brushSize * displayScale,
      lineCap: 'round',
      lineJoin: 'round',
      opacity: S.activeTool === 'eraser' ? 1 : 0.7,
      globalCompositeOperation: S.activeTool === 'eraser' ? 'destination-out' : 'source-over',
    });
    // Tag the stroke with the category active at draw time. commitAnchor
    // groups by this attribute so each category's strokes are committed as
    // their own mask -- prevents the bug where switching active category
    // before committing made every prior stroke take on the latest cat.
    currentStroke.setAttr('cat', S.activeCategory);
    if (S.activeTool === 'eraser') {
      // Eraser is destination-out; the cat tag here is just so removal
      // semantics travel with the stroke alongside other same-cat work.
      currentStroke.setAttr('isEraser', true);
    }
    editLayer.add(currentStroke);
    editLayer.batchDraw();
  } else if (S.activeTool === 'fill') {
    if (!S.activeCategory) { setStatus('pick an active category before filling'); return; }
    const op = stageToOriginal(pos);
    bucketFill(Math.round(op.x), Math.round(op.y));
  } else if (S.activeTool === 'sam-click') {
    const op = stageToOriginal(pos);
    const label = parseInt($('#point-label').value);
    S.samPoints.push({ x: op.x, y: op.y, label });
    const dot = new Konva.Circle({
      x: pos.x, y: pos.y, radius: 5,
      fill: label ? '#0aff50' : '#ff4040',
      stroke: 'white', strokeWidth: 1.5,
      listening: false,
    });
    promptLayer.add(dot);
    promptLayer.batchDraw();
    setStatus(`SAM point: ${S.samPoints.length} (pos=${label})`);
  } else if (S.activeTool === 'sam-box') {
    const op = stageToOriginal(pos);
    if (S.samBoxFirstClick === null) {
      S.samBoxFirstClick = op;
      setStatus(`SAM box: corner 1 at (${Math.round(op.x)}, ${Math.round(op.y)}). Click corner 2.`);
    } else {
      const c1 = S.samBoxFirstClick;
      S.samBox = [
        Math.min(c1.x, op.x), Math.min(c1.y, op.y),
        Math.max(c1.x, op.x), Math.max(c1.y, op.y),
      ];
      S.samBoxFirstClick = null;
      promptLayer.destroyChildren();
      const [x1,y1,x2,y2] = S.samBox;
      promptLayer.add(new Konva.Rect({
        x: x1 * displayScale, y: y1 * displayScale,
        width: (x2-x1) * displayScale, height: (y2-y1) * displayScale,
        stroke: '#0aff50', strokeWidth: 2, dash: [6,3], listening: false,
      }));
      promptLayer.batchDraw();
      setStatus(`SAM box set. Click Preview to run SAM3.`);
    }
  }
}
function onPointerMove(e) {
  if (!painting || !currentStroke) return;
  const pos = stage.getPointerPosition();
  if (!pos) return;
  const pts = currentStroke.points();
  currentStroke.points([...pts, pos.x, pos.y]);
  editLayer.batchDraw();
}
function onPointerUp(e) {
  if (painting) {
    painting = false;
    // Pre-stroke snapshot was pushed in onPointerDown; nothing to record here.
    currentStroke = null;
  }
}

// ---------- Edit-layer snapshot helper (used by undo + bucket fill) ----------
// Captures the current edit layer to a fresh canvas at original image
// resolution. Used to push a "before" snapshot onto the undo stack prior to
// any destructive operation.
function snapshotEditLayer() {
  const W = S.width, H = S.height;
  const cvs = document.createElement('canvas');
  cvs.width = W; cvs.height = H;
  const editCanvas = editLayer.toCanvas({ width: stage.width(), height: stage.height(), pixelRatio: 1 });
  cvs.getContext('2d').drawImage(editCanvas, 0, 0, W, H);
  return cvs;
}

const UNDO_LIMIT_FRAME = 30;
function pushEditLayerUndo(label) {
  // Capture the edit layer state *before* a mutation, push to the per-frame
  // undo stack. Caller mutates afterwards. Cap the stack to avoid unbounded
  // memory growth during long edit sessions on a single frame.
  if (!stage || !S.snippet) return;
  S.undoStack.push({ type: 'raster', canvas: snapshotEditLayer(), label: label || 'edit' });
  while (S.undoStack.length > UNDO_LIMIT_FRAME) S.undoStack.shift();
}

// ---------- Bucket fill ----------
// Flood-fill the empty region around (seedX, seedY) on the edit layer with
// the active category color. Operates at original image resolution so the
// fill is committed at full fidelity.
//
// Boundary threshold: the brush is drawn at opacity 0.7 (core alpha ~178)
// with antialiased edges. A low threshold (e.g. 20) catches the outer AA
// pixels and leaves a 1-2px gap between the fill and the visible stroke
// centre. Setting the threshold above the AA range (alpha >= 130) treats
// only the opaque stroke core as boundary, so the fill spreads into the
// soft edge and eliminates the gap.
const FILL_ALPHA_BOUNDARY = 130;
const FILL_OUTPUT_ALPHA = 220;

// Cross-category boundary helper.
// Returns a Uint8Array of length W*H whose value is 1 where any *other*
// category's mask has alpha above the boundary threshold (so the fill
// treats them as walls), 0 otherwise. Walks S.gt polygons + S.approved /
// S.propagated RLEs and renders them onto a single off-screen canvas at
// original image resolution.
function buildOtherCategoryBoundary(activeCategory) {
  const W = S.width, H = S.height;
  const cvs = document.createElement('canvas');
  cvs.width = W; cvs.height = H;
  const ctx = cvs.getContext('2d');
  ctx.fillStyle = 'rgba(255,255,255,1)';
  ctx.strokeStyle = 'rgba(255,255,255,1)';
  ctx.lineWidth = 2;

  // GT polygons (already in original image coords, not display-scaled)
  for (const [cat, polys] of Object.entries(S.gt || {})) {
    if (cat === activeCategory) continue;
    for (const flat of polys) {
      if (!flat || flat.length < 6) continue;
      ctx.beginPath();
      for (let i = 0; i < flat.length; i += 2) {
        const x = flat[i], y = flat[i + 1];
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
  }

  // Approved + propagated RLE masks for non-active categories
  const stampRle = (rle) => {
    try {
      const id = decodeRLE(rle);  // ImageData with white-on-transparent
      const t = document.createElement('canvas');
      t.width = id.width; t.height = id.height;
      t.getContext('2d').putImageData(id, 0, 0);
      ctx.drawImage(t, 0, 0, W, H);
    } catch (e) {
      console.warn('boundary RLE decode failed', e);
    }
  };
  for (const [cat, rle] of Object.entries(S.approved || {})) {
    if (cat === activeCategory) continue;
    stampRle(rle);
  }
  for (const [cat, rle] of Object.entries(S.propagated || {})) {
    if (cat === activeCategory) continue;
    stampRle(rle);
  }

  const od = ctx.getImageData(0, 0, W, H).data;
  const wall = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    if (od[i * 4 + 3] > FILL_ALPHA_BOUNDARY) wall[i] = 1;
  }
  return wall;
}

function bucketFill(seedX, seedY) {
  const W = S.width, H = S.height;
  if (seedX < 0 || seedX >= W || seedY < 0 || seedY >= H) return;

  // Render the current edit layer at original resolution for hit-testing
  const tmp = snapshotEditLayer();
  const tctx = tmp.getContext('2d');
  const imgData = tctx.getImageData(0, 0, W, H);
  const d = imgData.data;

  // Other-category masks (GT polygons + approved/propagated RLEs) act as
  // additional walls so a fill respects neighbouring tissue/tool boundaries
  // even when the user's own strokes don't fully enclose the region.
  const otherWall = buildOtherCategoryBoundary(S.activeCategory);

  const seedLin = seedY * W + seedX;
  const seedI = seedLin * 4;
  if (d[seedI + 3] > FILL_ALPHA_BOUNDARY) {
    setStatus('fill: seed is on a painted pixel — click inside an empty region');
    return;
  }
  if (otherWall[seedLin]) {
    setStatus('fill: seed is on a neighbouring mask (' +
              'click inside an empty region between boundaries)');
    return;
  }

  // Push pre-fill snapshot for undo
  pushEditLayerUndo('fill');

  // 4-connected BFS, alpha-thresholded against (edit-layer OR other-cat masks)
  const visited = new Uint8Array(W * H);
  const queue = new Int32Array(W * H);
  let head = 0, tail = 0;
  queue[tail++] = seedLin;
  visited[seedLin] = 2;  // 2 = filled
  let filled = 0;

  const isWall = (ni) => d[ni * 4 + 3] > FILL_ALPHA_BOUNDARY || otherWall[ni];

  while (head < tail) {
    const lin = queue[head++];
    filled++;
    const x = lin % W, y = (lin - x) / W;
    if (x + 1 < W) { const ni = lin + 1; if (!visited[ni]) { if (isWall(ni)) visited[ni] = 1; else { visited[ni] = 2; queue[tail++] = ni; } } }
    if (x - 1 >= 0) { const ni = lin - 1; if (!visited[ni]) { if (isWall(ni)) visited[ni] = 1; else { visited[ni] = 2; queue[tail++] = ni; } } }
    if (y + 1 < H) { const ni = lin + W; if (!visited[ni]) { if (isWall(ni)) visited[ni] = 1; else { visited[ni] = 2; queue[tail++] = ni; } } }
    if (y - 1 >= 0) { const ni = lin - W; if (!visited[ni]) { if (isWall(ni)) visited[ni] = 1; else { visited[ni] = 2; queue[tail++] = ni; } } }
  }

  // Safety: if the fill flooded > 90% of the canvas, the seed wasn't enclosed.
  // Prompt for confirmation before committing — common cause of "wrecked everything".
  const pct = (100 * filled) / (W * H);
  if (pct > 90) {
    if (!confirm(`Fill flooded ${pct.toFixed(0)}% of the frame — your boundary likely isn't closed. Apply anyway?`)) {
      setStatus('fill cancelled (boundary not closed)');
      return;
    }
  }

  // Paint filled pixels with active category color
  const col = hexToRgb(catColor(S.activeCategory));
  for (let i = 0; i < W * H; i++) {
    if (visited[i] === 2) {
      const di = i * 4;
      d[di] = col.r; d[di + 1] = col.g; d[di + 2] = col.b;
      d[di + 3] = FILL_OUTPUT_ALPHA;
    }
  }
  tctx.putImageData(imgData, 0, 0);

  // Replace ONLY this category's children with the filled raster -- preserve
  // any other-category strokes the user has on the edit layer. (Bucket fill
  // used to nuke the entire edit layer, taking other cats' work with it.)
  const activeCat = S.activeCategory;
  const toRemove = editLayer.getChildren().filter(c => c.getAttr('cat') === activeCat);
  toRemove.forEach(c => c.destroy());
  const filledImg = new Konva.Image({
    image: tmp,
    width: stage.width(),
    height: stage.height(),
  });
  filledImg.setAttr('cat', activeCat);
  editLayer.add(filledImg);
  editLayer.batchDraw();
  S.undoStack.push({ type: 'raster', canvas: undoCanvas });
  setStatus(`fill: +${filled}px (${pct.toFixed(1)}% of frame)`);
}

// ---------- Undo ----------
// Unified raster-snapshot undo. Every mutating op on the edit layer
// (brush stroke, eraser stroke, bucket fill, SAM3 preview, accept-text,
// load-GT-as-editable, clear-edit-layer) calls pushEditLayerUndo() with
// a "before" snapshot. Undo restores the most recent one.
function undo() {
  const last = S.undoStack.pop();
  if (!last || last.type !== 'raster' || !last.canvas) return;
  editLayer.destroyChildren();
  editLayer.add(new Konva.Image({
    image: last.canvas,
    width: stage.width(),
    height: stage.height(),
  }));
  editLayer.batchDraw();
}

// ---------- Commit: edit layer → PNG → POST ----------
// Walks the edit-layer children, groups them by their `cat` attribute (set at
// draw-time / fill-time / load-GT-time), and POSTs one anchor commit per
// category so each category's strokes are saved under the correct mask --
// not lumped into whatever the active category happens to be at commit time.
async function commitAnchor() {
  if (!S.snippet) { setStatus('no snippet'); return; }
  if (!S.activeCategory) { setStatus('pick an active category'); return; }

  const origW = S.width;
  const origH = S.height;

  // Group children by category. Untagged children (legacy paint or things we
  // didn't tag) fall through under the active category as a safety net.
  const byCat = new Map();
  for (const child of editLayer.getChildren()) {
    const cat = child.getAttr('cat') || S.activeCategory;
    if (!byCat.has(cat)) byCat.set(cat, []);
    byCat.get(cat).push(child);
  }

  if (byCat.size === 0) {
    setStatus('nothing on edit layer to commit');
    return;
  }

  // For each category, render ONLY its children on a hidden Konva layer, then
  // export to PNG and POST. This isolates per-category strokes regardless of
  // z-order on the edit layer. We render at display resolution then scale up.
  const stageW = stage.width();
  const stageH = stage.height();

  setStatus(`committing ${byCat.size} categor${byCat.size === 1 ? 'y' : 'ies'}…`);
  const results = [];
  let totalPixels = 0;

  for (const [cat, nodes] of byCat) {
    const isolation = new Konva.Layer({ listening: false });
    stage.add(isolation);
    try {
      for (const node of nodes) isolation.add(node.clone({ listening: false }));
      isolation.batchDraw();
      const isolationCanvas = isolation.toCanvas({ width: stageW, height: stageH, pixelRatio: 1 });
      const tmp = document.createElement('canvas');
      tmp.width = origW; tmp.height = origH;
      const tctx = tmp.getContext('2d');
      tctx.drawImage(isolationCanvas, 0, 0, origW, origH);
      const imgData = tctx.getImageData(0, 0, origW, origH);
      const d = imgData.data;
      let on = 0;
      for (let i = 0; i < d.length; i += 4) {
        const has = d[i + 3] > 20 ? 255 : 0;
        d[i] = has; d[i + 1] = has; d[i + 2] = has; d[i + 3] = has;
        if (has) on++;
      }
      if (on === 0) {
        // All strokes on this cat were eraser-only -- skip it; nothing to commit
        continue;
      }
      tctx.putImageData(imgData, 0, 0);
      const dataUrl = tmp.toDataURL('image/png');
      const r = await apiPost('/api/anchor/commit', {
        frame_idx: S.frameIdx,
        category: cat,
        mask_png_b64: dataUrl,
      });
      results.push(`${cat}=${r.pixels}px`);
      totalPixels += r.pixels || 0;
    } catch (e) {
      setStatus(`commit failed for ${cat}: ${e.message}`);
      isolation.destroy();
      return;
    }
    isolation.destroy();
  }

  if (results.length === 0) {
    setStatus('commit: nothing to save (eraser-only edit?)');
  } else {
    setStatus(`committed: ${results.join(', ')} (${totalPixels}px total)`);
  }
  // Refresh approved masks for this frame
  try {
    const m = await apiGet(`/api/masks/${S.frameIdx}`);
    S.approved = m.approved || {};
    S.propagated = m.propagated || {};
  } catch (_) {}
  editLayer.destroyChildren();
  editLayer.batchDraw();
  // Any GT we'd pulled onto the edit layer is now committed back into approved
  for (const cat of byCat.keys()) S.editingGT.delete(cat);
  renderOverlays();
  refreshMasksList();
}

// ---------- Preview (SAM3 click/box) ----------
async function previewSAM() {
  if (!S.snippet) return;
  if (!S.activeCategory) { setStatus('pick an active category'); return; }
  if (S.samPoints.length === 0 && !S.samBox) {
    setStatus('no SAM prompts placed');
    return;
  }
  setStatus('running SAM3 preview…');
  try {
    const body = {
      frame_idx: S.frameIdx,
      category: S.activeCategory,
      points: S.samPoints.map(p => [p.x, p.y, p.label]),
    };
    if (S.samBox) body.box = S.samBox;
    const r = await apiPost('/api/preview/click', body);
    // Render preview on the edit layer (so user can further refine + commit)
    const imgData = decodeRLE(r.rle);
    const col = hexToRgb(catColor(S.activeCategory));
    for (let i = 0; i < imgData.data.length; i += 4) {
      if (imgData.data[i+3] > 0) {
        imgData.data[i] = col.r;
        imgData.data[i+1] = col.g;
        imgData.data[i+2] = col.b;
        imgData.data[i+3] = 180;
      }
    }
    const cvs = document.createElement('canvas');
    cvs.width = imgData.width;
    cvs.height = imgData.height;
    cvs.getContext('2d').putImageData(imgData, 0, 0);
    pushEditLayerUndo('sam-preview');
    editLayer.destroyChildren();
    editLayer.add(new Konva.Image({
      image: cvs,
      width: stage.width(),
      height: stage.height(),
    }));
    editLayer.batchDraw();
    setStatus(`preview: ${r.pixels}px. Click Commit to keep, or erase/add brush to refine.`);
  } catch (e) {
    setStatus(`preview failed: ${e.message}`);
  }
}

// ---------- Text prompt ----------
async function runTextPrompt() {
  if (!S.snippet) return;
  if (!S.activeCategory) { setStatus('pick an active category'); return; }
  const text = $('#text-prompt').value.trim();
  if (!text) { setStatus('enter a text prompt'); return; }
  const conf = parseFloat($('#text-conf').value);
  setStatus(`running text prompt "${text}" @ conf≥${conf.toFixed(2)} (loads image model on first call ~20s)…`);
  try {
    const r = await apiPost('/api/preview/text', {
      frame_idx: S.frameIdx,
      category: S.activeCategory,
      text,
      conf_threshold: conf,
      max_results: 10,
    });
    S.textDets = (r.detections || []).map(d => ({ ...d, selected: false }));
    renderTextDetections();
    setStatus(`text returned ${r.detections.length}/${r.n_total} detections. Tick rows or click numbered overlays to select. Then Accept Selected.`);
  } catch (e) {
    setStatus(`text prompt failed: ${e.message}`);
  }
}

function renderTextDetections() {
  // Render all detections on edit layer with selection state encoded in opacity
  editLayer.destroyChildren();
  const list = $('#text-results');
  list.innerHTML = '';
  const col = hexToRgb(catColor(S.activeCategory));

  S.textDets.forEach((d, i) => {
    try {
      const imgData = decodeRLE(d.rle);
      const alpha = d.selected ? 200 : 100;
      const r = d.selected ? col.r : 160;
      const g = d.selected ? col.g : 160;
      const b = d.selected ? col.b : 160;
      for (let j = 0; j < imgData.data.length; j += 4) {
        if (imgData.data[j+3] > 0) {
          imgData.data[j] = r; imgData.data[j+1] = g; imgData.data[j+2] = b;
          imgData.data[j+3] = alpha;
        }
      }
      const cvs = document.createElement('canvas');
      cvs.width = imgData.width; cvs.height = imgData.height;
      cvs.getContext('2d').putImageData(imgData, 0, 0);
      const ki = new Konva.Image({
        image: cvs,
        width: stage.width(),
        height: stage.height(),
        listening: true,
      });
      ki.setAttr('detIdx', i);
      ki.on('click tap', () => toggleTextDetection(i));
      editLayer.add(ki);

      // Number + score badge
      if (d.box && d.box.length === 4) {
        const [x1, y1, x2, y2] = d.box;
        const cx = (x1 + x2) / 2 * displayScale;
        const cy = (y1 + y2) / 2 * displayScale;
        const tag = new Konva.Label({ x: cx, y: cy, listening: true });
        tag.add(
          new Konva.Tag({
            fill: d.selected ? 'rgba(0,180,216,0.85)' : 'rgba(0,0,0,0.7)',
            cornerRadius: 3,
          }),
          new Konva.Text({
            text: (d.selected ? '✓ ' : '') + `#${i} ${(d.score*100).toFixed(0)}%`,
            fontSize: 13, padding: 3, fill: 'white',
          })
        );
        tag.on('click tap', () => toggleTextDetection(i));
        editLayer.add(tag);
      }
    } catch (e) { console.error('det render', e); }

    // Sidebar list row
    const row = document.createElement('label');
    row.style.cssText = 'display:flex; align-items:center; gap:6px; padding:5px 6px; cursor:pointer; border-bottom:1px solid var(--border);';
    row.innerHTML = `
      <input type="checkbox" ${d.selected ? 'checked' : ''}>
      <span style="flex:1">#${i} <b>${(d.score*100).toFixed(1)}%</b> · ${d.pixels}px</span>
    `;
    row.onmouseover = () => row.style.background = 'var(--panel-2)';
    row.onmouseout = () => row.style.background = '';
    row.querySelector('input').onchange = () => toggleTextDetection(i);
    list.appendChild(row);
  });
  editLayer.batchDraw();
}

function toggleTextDetection(i) {
  if (!S.textDets[i]) return;
  S.textDets[i].selected = !S.textDets[i].selected;
  renderTextDetections();
}

function selectAllText(val) {
  S.textDets.forEach(d => { d.selected = val; });
  renderTextDetections();
}

function acceptSelectedTextDets() {
  const sel = S.textDets.filter(d => d.selected);
  if (sel.length === 0) {
    setStatus('no detections selected — tick at least one');
    return;
  }
  // Decode all selected RLEs and union them into a single binary mask
  const W = S.width, H = S.height;
  const union = new Uint8Array(W * H);
  for (const d of sel) {
    const imgData = decodeRLE(d.rle);
    const data = imgData.data;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const di = (y * W + x) * 4;
        if (data[di + 3] > 0) union[y * W + x] = 1;
      }
    }
  }
  // Render union as opaque edit layer mask
  const out = new Uint8ClampedArray(W * H * 4);
  const col = hexToRgb(catColor(S.activeCategory));
  for (let i = 0; i < W * H; i++) {
    if (union[i]) {
      const di = i * 4;
      out[di] = col.r; out[di+1] = col.g; out[di+2] = col.b; out[di+3] = 220;
    }
  }
  const imgData = new ImageData(out, W, H);
  const cvs = document.createElement('canvas');
  cvs.width = W; cvs.height = H;
  cvs.getContext('2d').putImageData(imgData, 0, 0);
  pushEditLayerUndo('accept-text');
  editLayer.destroyChildren();
  editLayer.add(new Konva.Image({
    image: cvs, width: stage.width(), height: stage.height(),
  }));
  editLayer.batchDraw();
  // Clear detection list
  S.textDets = [];
  $('#text-results').innerHTML = '';
  const px = union.reduce((s, v) => s + v, 0);
  setStatus(`accepted ${sel.length} detection(s) as union (${px}px) on edit layer. Refine with brush/eraser, then Commit.`);
}

// ---------- Propagate ----------
async function propagate() {
  if (!S.snippet) return;
  const n = parseInt($('#prop-n').value) || 120;
  setStatus(`propagating ${n} frames each direction…`);
  try {
    const r = await apiPost('/api/propagate', {
      frame_idx: S.frameIdx,
      max_frames_per_direction: n,
    });
    setStatus(`propagation done: ${r.frames_touched} frames in ${r.elapsed_s}s`);
    // Refresh current frame masks
    const m = await apiGet(`/api/masks/${S.frameIdx}`);
    S.propagated = m.propagated || {};
    S.approved = m.approved || {};
    renderOverlays();
    refreshMasksList();
  } catch (e) {
    setStatus(`propagate failed: ${e.message}`);
  }
}

// ---------- Export ----------
async function exportSnippet() {
  if (!S.snippet) return;
  setStatus('exporting…');
  try {
    const r = await apiPost('/api/export', {
      concat_tool_instances: $('#concat-tool').checked,
    });
    setStatus(`exported: ${r.path}\napproved frames: ${r.approved_frames}, propagated: ${r.propagated_frames}`);
  } catch (e) {
    setStatus(`export failed: ${e.message}`);
  }
}

// ---------- Load GT as editable ----------
async function loadGTEditable() {
  if (!S.snippet || !S.activeCategory) return;
  const polys = S.gt[S.activeCategory] || [];
  if (polys.length === 0) {
    setStatus(`no GT for ${S.activeCategory} at this frame`);
    return;
  }
  // Rasterize polygon to a single Konva.Image so brush+eraser fully edit it
  // (Konva.Line is a vector primitive; its stroke pixels are redrawn each frame
  //  and would persist through eraser strokes. A raster image is fully editable.)
  pushEditLayerUndo('load-gt');
  // Only clear the *active category* off the edit layer -- preserve any other
  // cats' strokes the user is mid-way through.
  const activeCat = S.activeCategory;
  editLayer.getChildren().filter(c => c.getAttr('cat') === activeCat)
                    .forEach(c => c.destroy());
  const W = S.width, H = S.height;
  const cvs = document.createElement('canvas');
  cvs.width = W; cvs.height = H;
  const ctx = cvs.getContext('2d');
  const col = catColor(activeCat);
  ctx.fillStyle = col + 'CC';  // ~80% alpha
  for (const flat of polys) {
    if (flat.length < 6) continue;  // need ≥3 points
    ctx.beginPath();
    for (let i = 0; i < flat.length; i += 2) {
      const x = flat[i], y = flat[i + 1];
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  }
  const gtImg = new Konva.Image({
    image: cvs,
    width: stage.width(),
    height: stage.height(),
  });
  gtImg.setAttr('cat', activeCat);
  editLayer.add(gtImg);
  editLayer.batchDraw();
  // Hide the GT outline for this category so it doesn't look un-erasable
  S.editingGT.add(S.activeCategory);
  renderOverlays();
  setStatus(`loaded ${polys.length} GT polygon(s) for ${S.activeCategory} as raster — brush/eraser fully editable. Commit when done.`);
}

// ---------- Undo / Redo ----------
async function undoAction() {
  // Unified undo: stroke-level first (covers brush AND eraser strokes on edit layer),
  // then backend mask-state undo (delete/commit/clear/propagate).
  if (S.undoStack && S.undoStack.length > 0) {
    undo();
    setStatus(`undo stroke · ${S.undoStack.length} stroke(s) remaining`);
    return;
  }
  if (!S.snippet) return;
  try {
    const r = await apiPost('/api/undo', {});
    setStatus(`undo: ${r.description} · ${r.restored} mask(s) restored · ${r.undo_remaining} undo / ${r.redo_available} redo`);
    await refreshAfterMaskChange(S.frameIdx);
  } catch (e) {
    setStatus(`undo: ${e.message}`);
  }
}

async function redoAction() {
  if (!S.snippet) return;
  try {
    const r = await apiPost('/api/redo', {});
    setStatus(`redo: ${r.description} · ${r.restored} mask(s) reapplied · ${r.undo_available} undo / ${r.redo_remaining} redo`);
    await refreshAfterMaskChange(S.frameIdx);
  } catch (e) {
    setStatus(`redo: ${e.message}`);
  }
}

// ---------- Master mask list ----------
let _masksItemsCache = [];

async function refreshMasksList() {
  if (!S.snippet) return;
  try {
    const r = await apiGet('/api/masks/list');
    _masksItemsCache = r.items || [];
    renderMasksList();  // sets counter based on current filter scope
  } catch (e) {
    setStatus(`mask-list refresh failed: ${e.message}`);
  }
}

function renderMasksList() {
  const list = $('#masks-list');
  list.innerHTML = '';
  const showGt = $('#filter-gt')?.checked ?? true;
  const showAp = $('#filter-approved')?.checked ?? true;
  const showPr = $('#filter-propagated')?.checked ?? true;
  const currOnly = $('#filter-current')?.checked ?? true;
  const q = ($('#masks-search')?.value || '').trim().toLowerCase();

  const filtered = _masksItemsCache.filter(it => {
    if (currOnly && it.frame_idx !== S.frameIdx) return false;
    if (it.kind === 'gt' && !showGt) return false;
    if (it.kind === 'approved' && !showAp) return false;
    if (it.kind === 'propagated' && !showPr) return false;
    if (q && !it.category.toLowerCase().includes(q)) return false;
    return true;
  });

  // Update counter to reflect filter scope
  const fcounts = { gt: 0, approved: 0, propagated: 0 };
  for (const it of filtered) fcounts[it.kind] = (fcounts[it.kind] || 0) + 1;
  const scopeLabel = currOnly ? `frame ${S.frameIdx}` : 'all';
  $('#masks-counts').textContent =
    `[${scopeLabel}] GT:${fcounts.gt} A:${fcounts.approved} P:${fcounts.propagated}`;

  if (filtered.length === 0) {
    list.innerHTML = '<div style="padding:6px; color:var(--text-dim);">no masks match filter</div>';
    return;
  }

  // Cap render to avoid DOM overload (~960 GT rows on E_3 snippets)
  const CAP = 500;
  const shown = filtered.slice(0, CAP);

  for (const it of shown) {
    const row = document.createElement('div');
    const col = catColor(it.category);
    const isCurr = it.frame_idx === S.frameIdx;
    row.style.cssText = `padding:3px 6px; cursor:pointer; border-bottom:1px solid var(--border); ${isCurr ? 'background:var(--panel-2);' : ''}`;
    const kindMark = it.kind === 'approved' ? '●' : it.kind === 'propagated' ? '○' : '◆';
    const kindColor = it.kind === 'approved' ? 'var(--text)'
                    : it.kind === 'gt' ? '#ffd60a'
                    : 'var(--text-dim)';
    const fnum = it.frame_num !== null && it.frame_num !== undefined ? ` #${it.frame_num}` : '';
    const px = it.kind === 'gt' ? `${it.pixels}pt` : `${it.pixels}px`;
    row.innerHTML = `
      <span style="display:inline-block;width:8px;height:8px;background:${col};border-radius:2px;margin-right:5px"></span>
      <span style="color:${kindColor}">${kindMark}</span>
      f${it.frame_idx}${fnum} · ${it.category} · ${px}
    `;
    row.title = `${it.kind} · frame_idx ${it.frame_idx} · ${it.category}`;
    row.onmouseover = () => row.style.background = 'var(--accent-glow)';
    row.onmouseout = () => row.style.background = isCurr ? 'var(--accent-glow)' : '';
    row.onclick = () => loadFrame(it.frame_idx);
    row.oncontextmenu = (e) => showMaskContextMenu(e, it);
    list.appendChild(row);
  }
  if (filtered.length > CAP) {
    const more = document.createElement('div');
    more.style.cssText = 'padding:6px; color:var(--text-dim); text-align:center;';
    more.textContent = `…${filtered.length - CAP} more (use category filter to narrow)`;
    list.appendChild(more);
  }
}

// ---------- Context menu ----------
function hideContextMenu() {
  document.querySelectorAll('.ctx-menu').forEach(el => el.remove());
}

function showMaskContextMenu(e, item) {
  e.preventDefault();
  e.stopPropagation();
  hideContextMenu();
  const menu = document.createElement('div');
  menu.className = 'ctx-menu';
  menu.style.left = e.clientX + 'px';
  menu.style.top = e.clientY + 'px';

  const header = document.createElement('div');
  header.className = 'ctx-header';
  header.textContent = `${item.kind} · ${item.category} · f${item.frame_idx}`;
  menu.appendChild(header);

  const sep = document.createElement('div');
  sep.className = 'ctx-divider';
  menu.appendChild(sep);

  const addItem = (label, fn, danger = false) => {
    const it = document.createElement('div');
    it.className = 'ctx-item' + (danger ? ' danger' : '');
    it.textContent = label;
    it.onclick = (ev) => { ev.stopPropagation(); hideContextMenu(); fn(); };
    menu.appendChild(it);
  };

  addItem(`Jump to frame ${item.frame_idx}`, () => loadFrame(item.frame_idx));

  if (item.kind === 'gt') {
    addItem('Load as editable here', () => {
      loadFrame(item.frame_idx).then(() => {
        S.activeCategory = item.category;
        const acDd = $('#active-cat'); if (acDd) acDd.value = item.category;
        renderCategoryList();
        loadGTEditable();
      });
    });
    addItem('GT is read-only', () => {}, true);
  } else {
    addItem(`Delete this ${item.kind} mask`, () => deleteMask(item), true);
    addItem(`Delete all ${item.kind} on frame ${item.frame_idx}`,
            () => clearMasks({ kind: item.kind, frame_idx: item.frame_idx }), true);
    addItem(`Delete all ${item.kind} for ${item.category}`,
            () => clearMasks({ kind: item.kind, category: item.category }), true);
  }

  document.body.appendChild(menu);
  // Close on next outside click / Escape
  setTimeout(() => {
    document.addEventListener('click', hideContextMenu, { once: true });
    document.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') hideContextMenu();
    }, { once: true });
  }, 0);
}

async function deleteMask(item) {
  try {
    await apiPost('/api/masks/delete', {
      frame_idx: item.frame_idx,
      category: item.category,
      kind: item.kind,
    });
    setStatus(`deleted ${item.kind} ${item.category} @ f${item.frame_idx}`);
    await refreshAfterMaskChange(item.frame_idx);
  } catch (e) {
    setStatus(`delete failed: ${e.message}`);
  }
}

async function clearMasks(opts) {
  const sure = confirm(
    `Delete ${opts.kind || 'all'} masks` +
    (opts.category ? ` for ${opts.category}` : '') +
    (opts.frame_idx !== undefined ? ` on frame ${opts.frame_idx}` : ' across all frames') +
    '?'
  );
  if (!sure) return;
  try {
    const r = await apiPost('/api/masks/clear', opts);
    setStatus(`deleted ${r.deleted} mask(s)`);
    await refreshAfterMaskChange(S.frameIdx);
  } catch (e) {
    setStatus(`clear failed: ${e.message}`);
  }
}

async function refreshAfterMaskChange(fidx) {
  await refreshMasksList();
  // Re-pull current frame masks
  try {
    const m = await apiGet(`/api/masks/${S.frameIdx}`);
    S.approved = m.approved || {};
    S.propagated = m.propagated || {};
    renderOverlays();
  } catch (_) {}
}

// ---------- UI binding ----------
function setStatus(msg) {
  $('#status-box').textContent = msg;
  console.log('[status]', msg);
}

function renderCategoryList() {
  const list = $('#cat-list');
  list.innerHTML = '';
  for (const cat of S.categories) {
    const el = document.createElement('label');
    const col = catColor(cat);
    el.innerHTML = `
      <input type="radio" name="active-cat" value="${cat}" ${cat===S.activeCategory?'checked':''}>
      <span style="display:inline-block;width:10px;height:10px;background:${col};border-radius:2px;"></span>
      ${cat}
    `;
    el.querySelector('input').onchange = () => {
      S.activeCategory = cat;
      $('#active-cat').value = cat;
    };
    list.appendChild(el);
  }
  // populate active-cat dropdown too
  const acDd = $('#active-cat');
  acDd.innerHTML = '';
  for (const cat of S.categories) {
    const o = document.createElement('option');
    o.value = cat; o.textContent = cat;
    acDd.appendChild(o);
  }
  if (S.activeCategory) acDd.value = S.activeCategory;
}

async function refreshEpisodes() {
  const r = await apiGet('/api/episodes');
  const dd = $('#episode-dd');
  dd.innerHTML = '';
  for (const ep of r.episodes) {
    const o = document.createElement('option');
    o.value = ep; o.textContent = ep;
    dd.appendChild(o);
  }
  if (r.episodes.length > 0) {
    S.episode = r.episodes[0];
    dd.value = S.episode;
    await refreshSnippets();
  }
}
async function refreshSnippets() {
  if (!S.episode) return;
  const r = await apiGet(`/api/episodes/${S.episode}/snippets`);
  const dd = $('#snippet-dd');
  dd.innerHTML = '';
  for (const s of r.snippets) {
    const o = document.createElement('option');
    o.value = s.snippet_id;
    o.textContent = `${s.snippet_id} (${s.num_frames}f)`;
    dd.appendChild(o);
  }
  if (r.snippets.length > 0) dd.value = r.snippets[0].snippet_id;
}
function applyLightweightUI() {
  // Disable the Propagate button + slider when in lightweight mode.
  const btn = $('#prop-btn');
  const slider = $('#prop-n');
  if (!btn) return;
  if (S.lightweight) {
    btn.disabled = true;
    btn.textContent = 'Propagate (disabled)';
    btn.title = 'Lightweight mode: anchors are saved only. Use scripts/propagate_from_autosave.py on a stronger GPU.';
    if (slider) slider.disabled = true;
  } else {
    btn.disabled = false;
    btn.textContent = 'Propagate';
    btn.title = '';
    if (slider) slider.disabled = false;
  }
}

async function loadSnippet() {
  const ep = $('#episode-dd').value;
  const sid = $('#snippet-dd').value;
  if (!ep || !sid) return;
  setStatus(`loading ${ep}/${sid}…`);
  try {
    const r = await apiPost('/api/session/open', {
      episode: ep,
      snippet_id: sid,
      categories: ['Tool', 'Liver', 'Gallbladder'],
    });
    S.episode = r.episode;
    S.snippet = r.snippet_id;
    S.nFrames = r.n_frames;
    S.width = r.width;
    S.height = r.height;
    S.splitSize = r.split_size;
    S.framesDirName = r.frames_dir_name || 'frames_left';
    S.startFrame = r.start_frame;
    S.endFrame = r.end_frame;
    S.categories = r.categories;
    S.activeCategory = S.categories[0];
    S.lightweight = !!r.lightweight;
    applyLightweightUI();

    $('#frame-slider').max = Math.max(0, S.nFrames - 1);
    $('#frame-slider').value = 0;
    initKonva(S.width, S.height);
    renderCategoryList();
    await loadFrame(0);
    refreshMasksList();
    const lwTag = S.lightweight ? ' [LIGHTWEIGHT — propagate disabled]' : '';
    setStatus(`loaded ${r.episode}/${r.snippet_id} · ${r.n_frames} frames · restored ${r.restored_anchors} anchors${lwTag}`);
  } catch (e) {
    setStatus(`load failed: ${e.message}`);
  }
}

function bindEvents() {
  $('#episode-dd').onchange = () => { S.episode = $('#episode-dd').value; refreshSnippets(); };
  $('#load-btn').onclick = loadSnippet;
  $('#add-cat').onclick = () => {
    const name = $('#new-cat').value.trim();
    if (!name) return;
    if (!S.categories.includes(name)) {
      S.categories.push(name);
      renderCategoryList();
      apiPost('/api/categories', { categories: S.categories });
    }
    $('#new-cat').value = '';
  };
  $('#active-cat').onchange = () => {
    S.activeCategory = $('#active-cat').value;
    renderCategoryList();
  };

  $('#prev-btn').onclick = () => loadFrame(Math.max(0, S.frameIdx - 1));
  $('#next-btn').onclick = () => loadFrame(Math.min(S.nFrames - 1, S.frameIdx + 1));
  $('#frame-slider').oninput = (e) => loadFrame(parseInt(e.target.value));

  document.querySelectorAll('.tool-btn').forEach(b => {
    b.onclick = () => {
      document.querySelectorAll('.tool-btn').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      S.activeTool = b.dataset.tool;
      promptLayer.destroyChildren();
      S.samPoints = []; S.samBox = null; S.samBoxFirstClick = null;
      promptLayer.batchDraw();
      // Show/hide text prompt panel
      $('#text-prompt-group').style.display = (S.activeTool === 'sam-text') ? '' : 'none';
      setStatus(`tool: ${S.activeTool}`);
    };
  });

  $('#text-conf').oninput = (e) => {
    $('#text-conf-val').textContent = parseFloat(e.target.value).toFixed(2);
  };
  $('#text-run-btn').onclick = runTextPrompt;
  $('#text-accept-btn').onclick = acceptSelectedTextDets;
  $('#text-select-all').onclick = () => selectAllText(true);
  $('#text-select-none').onclick = () => selectAllText(false);

  $('#brush-size').oninput = (e) => {
    S.brushSize = parseInt(e.target.value);
    $('#brush-size-val').textContent = `${S.brushSize}px`;
  };
  $('#point-label').onchange = (e) => { S.pointLabel = parseInt(e.target.value); };

  $('#preview-btn').onclick = previewSAM;
  $('#commit-btn').onclick = commitAnchor;
  $('#clear-btn').onclick = () => {
    // Snapshot pre-clear so undo can restore the cleared edit layer
    pushEditLayerUndo('clear-edit');
    editLayer.destroyChildren();
    editLayer.batchDraw();
    promptLayer.destroyChildren();
    S.samPoints = []; S.samBox = null; S.samBoxFirstClick = null;
    promptLayer.batchDraw();
    // Note: undoStack is intentionally NOT cleared — we just pushed the
    // pre-clear snapshot, so Ctrl+Z still restores it.
    S.editingGT.clear();
    renderOverlays();
    setStatus('edit layer cleared (Ctrl+Z to undo)');
  };
  $('#load-gt-btn').onclick = loadGTEditable;

  $('#prop-btn').onclick = propagate;
  $('#export-btn').onclick = exportSnippet;
  $('#masks-refresh').onclick = refreshMasksList;
  ['#filter-gt', '#filter-approved', '#filter-propagated', '#filter-current'].forEach(sel => {
    $(sel).onchange = renderMasksList;
  });
  $('#masks-search').oninput = renderMasksList;
  $('#clear-prop-btn').onclick = () => clearMasks({ kind: 'propagated' });
  $('#clear-app-btn').onclick = () => clearMasks({ kind: 'approved' });
  $('#undo-btn').onclick = undoAction;
  $('#redo-btn').onclick = redoAction;

  $('#show-gt').onchange = e => { S.showGt = e.target.checked; renderOverlays(); };
  $('#show-approved').onchange = e => { S.showApproved = e.target.checked; renderOverlays(); };
  $('#show-propagated').onchange = e => { S.showPropagated = e.target.checked; renderOverlays(); };

  // Keyboard
  window.addEventListener('keydown', (e) => {
    if (document.activeElement && ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return;
    if (e.key === 'b') selectTool('brush');
    else if (e.key === 'e') selectTool('eraser');
    else if (e.key === 'f') selectTool('fill');
    else if (e.key === 's' && !e.shiftKey) selectTool('sam-click');
    else if (e.key === 'S' && e.shiftKey) selectTool('sam-box');
    else if (e.key === 't') selectTool('sam-text');
    else if (e.key === 'g') loadGTEditable();
    else if (e.key === 'Enter') { e.preventDefault(); commitAnchor(); }
    else if (e.key === ' ') {
      e.preventDefault();
      if (S.lightweight) {
        setStatus('lightweight mode: propagate disabled. Save anchors and replay on a stronger GPU.');
      } else {
        propagate();
      }
    }
    else if (e.key === 'z' && (e.ctrlKey || e.metaKey) && e.shiftKey) { e.preventDefault(); undoAction(); }
    else if (e.key === 'Z' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); undoAction(); }
    else if (e.key === 'y' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); redoAction(); }
    else if (e.key === 'z' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); undo(); }
    else if (e.key === '[') { $('#brush-size').value = Math.max(1, S.brushSize - 2); $('#brush-size').dispatchEvent(new Event('input')); }
    else if (e.key === ']') { $('#brush-size').value = Math.min(80, S.brushSize + 2); $('#brush-size').dispatchEvent(new Event('input')); }
    else if (e.key === 'ArrowLeft') { const step = e.shiftKey ? 10 : 1; loadFrame(Math.max(0, S.frameIdx - step)); }
    else if (e.key === 'ArrowRight') { const step = e.shiftKey ? 10 : 1; loadFrame(Math.min(S.nFrames - 1, S.frameIdx + step)); }
    else if (e.key === 'PageUp') { e.preventDefault(); loadFrame(findPrevKeyframe()); }
    else if (e.key === 'PageDown') { e.preventDefault(); loadFrame(findNextKeyframe()); }
    else if (e.key === 'Home') { e.preventDefault(); loadFrame(0); }
    else if (e.key === 'End') { e.preventDefault(); loadFrame(S.nFrames - 1); }
  });
}

// ---------- Keyframe navigation ----------
// A keyframe is a frame whose absolute video number is a multiple of the
// snippet's split_size. Cluster-format snippets have no split layout
// (S.splitSize is 0/null) -- in that case PageUp/PageDown act as ±1 frame.
function isKeyframeAt(idx) {
  if (!S.splitSize || S.splitSize <= 0) return false;
  const frameNum = S.startFrame + idx;
  return frameNum % S.splitSize === 0;
}
function findPrevKeyframe() {
  if (!S.splitSize || S.splitSize <= 0) return Math.max(0, S.frameIdx - 1);
  for (let i = S.frameIdx - 1; i >= 0; i--) {
    if (isKeyframeAt(i)) return i;
  }
  return 0;
}
function findNextKeyframe() {
  if (!S.splitSize || S.splitSize <= 0) return Math.min(S.nFrames - 1, S.frameIdx + 1);
  for (let i = S.frameIdx + 1; i < S.nFrames; i++) {
    if (isKeyframeAt(i)) return i;
  }
  return S.nFrames - 1;
}
function selectTool(name) {
  const b = document.querySelector(`.tool-btn[data-tool="${name}"]`);
  if (b) b.click();
}

// ---------- WebSocket heartbeat ----------
function startWS() {
  try {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'hello' }));
      setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'ping' }));
      }, 20000);
    };
    ws.onclose = () => setTimeout(startWS, 5000);
  } catch (e) {
    console.error('ws failed', e);
  }
}

// ---------- Health indicator ----------
async function refreshHealth() {
  try {
    const r = await apiGet('/api/health');
    const el = $('#app-health');
    if (!el) return;
    const ver = $('#app-version');
    if (ver && r.version) ver.textContent = `v${r.version}`;

    let cls = 'ok', glyph = '●', txt;
    if (r.no_model) { cls = 'warn'; txt = 'no-model'; }
    else if (!r.cuda_available) { cls = 'warn'; txt = 'cpu'; }
    else if (r.gpu && r.gpu.vram_pct_used > 90) {
      cls = 'err';
      txt = `${r.gpu.vram_pct_used.toFixed(0)}% vram · ${r.gpu.name}`;
    } else if (r.gpu) {
      txt = `${r.gpu.name.replace('NVIDIA ', '')} · ${r.gpu.vram_free_mb}/${r.gpu.vram_total_mb}MB`;
    } else {
      cls = 'warn'; txt = 'no gpu info';
    }
    el.innerHTML = `<span class="indicator ${cls}">${glyph}</span> ${txt}`;
  } catch (e) {
    const el = $('#app-health');
    if (el) el.innerHTML = `<span class="indicator err">●</span> health unreachable`;
  }
}

// ---------- Boot ----------
(async () => {
  bindEvents();
  await refreshEpisodes();
  startWS();
  refreshHealth();
  setInterval(refreshHealth, 10000);
  // Viewport resize → re-fit Konva stage
  window.addEventListener('resize', handleResize);
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(handleResize);
    ro.observe(document.getElementById('canvas-wrap'));
  }
  setStatus('select episode + snippet, then Load Snippet.');
})();
