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
  splitSize: 120,
  startFrame: 0,
  endFrame: 0,
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
  // undo stack: array of {type, payload}
  undoStack: [],
};

// ---------- DOM ----------
const $ = (sel) => document.querySelector(sel);

// ---------- Konva setup ----------
let stage, imageLayer, gtLayer, approvedLayer, propagatedLayer, editLayer, promptLayer;
let displayScale = 1;  // canvas display size / original size

function initKonva(width, height) {
  // Fit the canvas within the central area
  const wrap = $('#canvas-wrap');
  const maxW = wrap.clientWidth - 28;
  const maxH = wrap.clientHeight - 28;
  displayScale = Math.min(maxW / width, maxH / height, 1);
  const dw = Math.round(width * displayScale);
  const dh = Math.round(height * displayScale);

  $('#konva-container').innerHTML = '';
  stage = new Konva.Stage({
    container: 'konva-container',
    width: dw,
    height: dh,
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
}

function updateFrameInfo() {
  if (!S.snippet) return;
  const frameNum = S.startFrame + S.frameIdx;
  const split = Math.floor(frameNum / S.splitSize);
  const offset = frameNum % S.splitSize;
  const isKf = offset === 0 ? ' [KF]' : '';
  $('#frame-info').textContent =
    `Frame ${S.frameIdx + 1}/${S.nFrames} · #${frameNum} · split ${split} offset ${offset}${isKf}`;
}

// ---------- Overlays ----------
function renderOverlays() {
  gtLayer.destroyChildren();
  approvedLayer.destroyChildren();
  propagatedLayer.destroyChildren();

  if (S.showGt) {
    for (const [cat, polys] of Object.entries(S.gt)) {
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
    editLayer.add(currentStroke);
    editLayer.batchDraw();
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
    S.undoStack.push({ type: 'stroke', node: currentStroke });
    currentStroke = null;
  }
}

// ---------- Undo ----------
function undo() {
  const last = S.undoStack.pop();
  if (!last) return;
  if (last.type === 'stroke' && last.node) {
    last.node.destroy();
    editLayer.batchDraw();
  }
}

// ---------- Commit: edit layer → PNG → POST ----------
async function commitAnchor() {
  if (!S.snippet) { setStatus('no snippet'); return; }
  if (!S.activeCategory) { setStatus('pick an active category'); return; }

  // Render ONLY the edit layer as an opaque mask at original resolution.
  // We draw the edit layer onto a new canvas, then threshold alpha.
  const origW = S.width;
  const origH = S.height;
  const tmp = document.createElement('canvas');
  tmp.width = origW;
  tmp.height = origH;
  const tctx = tmp.getContext('2d');
  // The edit layer canvas is at display resolution; scale up.
  const editCanvas = editLayer.toCanvas({ width: stage.width(), height: stage.height(), pixelRatio: 1 });
  tctx.drawImage(editCanvas, 0, 0, origW, origH);
  // Threshold alpha to produce a binary mask (white where drawn)
  const imgData = tctx.getImageData(0, 0, origW, origH);
  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    const has = d[i+3] > 20 ? 255 : 0;
    d[i] = has; d[i+1] = has; d[i+2] = has; d[i+3] = has;
  }
  tctx.putImageData(imgData, 0, 0);
  const dataUrl = tmp.toDataURL('image/png');

  setStatus('committing anchor…');
  try {
    const r = await apiPost('/api/anchor/commit', {
      frame_idx: S.frameIdx,
      category: S.activeCategory,
      mask_png_b64: dataUrl,
    });
    setStatus(`anchor committed: ${r.pixels}px ${S.activeCategory}`);
    // Refresh approved masks for this frame
    const m = await apiGet(`/api/masks/${S.frameIdx}`);
    S.approved = m.approved || {};
    S.propagated = m.propagated || {};
    editLayer.destroyChildren();
    editLayer.batchDraw();
    renderOverlays();
  } catch (e) {
    setStatus(`commit failed: ${e.message}`);
  }
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
  editLayer.destroyChildren();
  const col = catColor(S.activeCategory);
  for (const flat of polys) {
    const pts = flat.map(v => v * displayScale);
    editLayer.add(new Konva.Line({
      points: pts, stroke: col, strokeWidth: 2,
      closed: true, fill: col + '80',
    }));
  }
  editLayer.batchDraw();
  setStatus(`loaded ${polys.length} GT polygons for ${S.activeCategory}. Brush/eraser to refine, then Commit.`);
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
    S.startFrame = r.start_frame;
    S.endFrame = r.end_frame;
    S.categories = r.categories;
    S.activeCategory = S.categories[0];

    $('#frame-slider').max = Math.max(0, S.nFrames - 1);
    $('#frame-slider').value = 0;
    initKonva(S.width, S.height);
    renderCategoryList();
    await loadFrame(0);
    setStatus(`loaded ${r.episode}/${r.snippet_id} · ${r.n_frames} frames · restored ${r.restored_anchors} anchors`);
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
      setStatus(`tool: ${S.activeTool}`);
    };
  });

  $('#brush-size').oninput = (e) => {
    S.brushSize = parseInt(e.target.value);
    $('#brush-size-val').textContent = `${S.brushSize}px`;
  };
  $('#point-label').onchange = (e) => { S.pointLabel = parseInt(e.target.value); };

  $('#preview-btn').onclick = previewSAM;
  $('#commit-btn').onclick = commitAnchor;
  $('#clear-btn').onclick = () => {
    editLayer.destroyChildren();
    editLayer.batchDraw();
    promptLayer.destroyChildren();
    S.samPoints = []; S.samBox = null; S.samBoxFirstClick = null;
    promptLayer.batchDraw();
    S.undoStack = [];
    setStatus('edit layer cleared');
  };
  $('#load-gt-btn').onclick = loadGTEditable;

  $('#prop-btn').onclick = propagate;
  $('#export-btn').onclick = exportSnippet;

  $('#show-gt').onchange = e => { S.showGt = e.target.checked; renderOverlays(); };
  $('#show-approved').onchange = e => { S.showApproved = e.target.checked; renderOverlays(); };
  $('#show-propagated').onchange = e => { S.showPropagated = e.target.checked; renderOverlays(); };

  // Keyboard
  window.addEventListener('keydown', (e) => {
    if (document.activeElement && ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return;
    if (e.key === 'b') selectTool('brush');
    else if (e.key === 'e') selectTool('eraser');
    else if (e.key === 's' && !e.shiftKey) selectTool('sam-click');
    else if (e.key === 'S' && e.shiftKey) selectTool('sam-box');
    else if (e.key === 'g') loadGTEditable();
    else if (e.key === 'Enter') { e.preventDefault(); commitAnchor(); }
    else if (e.key === ' ') { e.preventDefault(); propagate(); }
    else if (e.key === 'z' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); undo(); }
    else if (e.key === '[') { $('#brush-size').value = Math.max(1, S.brushSize - 2); $('#brush-size').dispatchEvent(new Event('input')); }
    else if (e.key === ']') { $('#brush-size').value = Math.min(80, S.brushSize + 2); $('#brush-size').dispatchEvent(new Event('input')); }
    else if (e.key === 'ArrowLeft') { const step = e.shiftKey ? 10 : 1; loadFrame(Math.max(0, S.frameIdx - step)); }
    else if (e.key === 'ArrowRight') { const step = e.shiftKey ? 10 : 1; loadFrame(Math.min(S.nFrames - 1, S.frameIdx + step)); }
  });
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

// ---------- Boot ----------
(async () => {
  bindEvents();
  await refreshEpisodes();
  startWS();
  setStatus('select episode + snippet, then Load Snippet.');
})();
