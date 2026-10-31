import { app } from "/scripts/app.js";

// Interactive focal-plane curve, drawn straight onto the node canvas (no DOM widget: the
// multiline keyframes textarea would otherwise swallow the free space and hide the curve).
// Drag points to aim the plane, click to add, right-click to delete. Every edit is written back
// into the `keyframes` widget, so the saved workflow stays the single source of truth.

const BG = "#17171b";
const BORDER = "#3a3a42";
const GRID = "#2a2a31";
const GRID2 = "#3c3c45";
const CURVE = "#ffd21e";
const POINT = "#4bd06d";
const POINT_HOT = "#8bffa9";
const GHOST = "#7d7f88";
const TEXT = "#c9ccd3";
const ACCENT = "#e69a1a";
const H = 168;
const PAD_L = 30;
const PAD_R = 8;
const PAD_T = 8;
const PAD_B = 16;
const HANDLE_R = 5;
const HIT_R = 10;

const clamp01 = (v) => Math.min(1, Math.max(0, v));

function parseKeys(text) {
  const keys = [];
  for (const raw of String(text || "").split(/[;\n]+/)) {
    const chunk = raw.trim();
    if (!chunk) continue;
    const i = chunk.indexOf(":");
    if (i < 0) continue;
    const f = chunk.slice(0, i).trim().toLowerCase();
    const t = chunk.slice(i + 1).trim();
    let value = null;
    if (/^-?\d+(\.\d+)?$/.test(t)) {
      value = parseFloat(t);
      if (value > 1) value /= 255;
    }
    keys.push({ raw: f, target: t, value: value === null ? null : clamp01(value) });
  }
  return keys;
}

const frameOf = (k, n) => (k.raw === "end" ? n - 1 : Math.min(n - 1, Math.max(0, parseInt(k.raw, 10) || 0)));
const serialize = (keys) => keys.map((k) => `${k.raw}:${k.target}`).join("; ");

// Mirror of a2b/prepare.py::parse_focus_keyframes: hold before/after, cosine/linear/step between.
function sampleCurve(keys, n, interp) {
  const pts = keys.map((k) => ({ f: frameOf(k, n), v: k.value })).sort((a, b) => a.f - b.f);
  const out = new Array(n).fill(null);
  const known = pts.filter((p) => p.v !== null);
  if (!known.length) return out;
  if (known.length === 1) return out.fill(known[0].v);
  for (let f = 0; f < n; f++) {
    if (f <= known[0].f) { out[f] = known[0].v; continue; }
    if (f >= known[known.length - 1].f) { out[f] = known[known.length - 1].v; continue; }
    let hi = 1;
    while (hi < known.length && known[hi].f < f) hi++;
    const a = known[hi - 1], b = known[hi];
    if (b.f === a.f) { out[f] = b.v; continue; }
    let t = (f - a.f) / (b.f - a.f);
    if (interp === "hold") t = 0;
    else if (interp === "smooth") t = 0.5 - 0.5 * Math.cos(Math.PI * t);
    out[f] = a.v + (b.v - a.v) * t;
  }
  return out;
}

app.registerExtension({
  name: "AnyToBokeh.FocusCurveEditor",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AnyToBokehFocusControl") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      attachCurve(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      setTimeout(() => this.setDirtyCanvas?.(true, true), 0);
      return r;
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function () {
      const r = onResize ? onResize.apply(this, arguments) : undefined;
      this.setDirtyCanvas?.(true, true);
      return r;
    };
  },
});

function attachCurve(node) {
  const kw = node.widgets?.find((w) => w.name === "keyframes");
  const iw = node.widgets?.find((w) => w.name === "interpolation");
  const fw = node.widgets?.find((w) => w.name === "clip_frames");
  if (!kw) return;

  const st = { hover: -1, drag: -1, area: null, keys: parseKeys(kw.value) };
  node._a2b = st;

  const n = () => Math.max(2, Number(fw?.value) || 240);
  const interp = () => iw?.value || "linear";

  function draw(ctx, node, width, y, height) {
    if (st.text !== kw.value) {          // value arrives later (configure/typing): re-parse once
      st.text = kw.value;
      st.keys = parseKeys(kw.value);
    }
    // the layout only hands us ~20-40 px for an unknown widget type, so ignore `height`
    // and draw a fixed box; the node is sized tall enough to hold it
    const a = { x: 8, y: y + 2, w: Math.max(80, width - 16), h: H };
    st.area = a;
    const keys = st.keys;
    const nf = n();

    ctx.save();
    ctx.fillStyle = BG;
    ctx.strokeStyle = BORDER;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(a.x, a.y, a.w, a.h);
    ctx.fill();
    ctx.stroke();

    const xOf = (f) => a.x + (f / Math.max(1, nf - 1)) * a.w;
    const yOf = (v) => a.y + (1 - v) * a.h;

    for (let i = 0; i <= 4; i++) {
      ctx.strokeStyle = i % 2 === 0 ? GRID2 : GRID;
      const gx = a.x + (i / 4) * a.w;
      ctx.beginPath(); ctx.moveTo(gx, a.y); ctx.lineTo(gx, a.y + a.h); ctx.stroke();
      const gy = a.y + (i / 4) * a.h;
      ctx.beginPath(); ctx.moveTo(a.x, gy); ctx.lineTo(a.x + a.w, gy); ctx.stroke();
    }

    ctx.fillStyle = TEXT;
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillStyle = "#7d7f88";
    ctx.font = "9px monospace";
    ctx.fillText("near", a.x + 4, a.y + 11);
    ctx.fillText("far", a.x + 4, a.y + a.h - 16);
    ctx.fillStyle = TEXT;
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText("0", a.x + 2, a.y + a.h - 4);
    ctx.textAlign = "right";
    ctx.fillText(String(nf - 1), a.x + a.w - 2, a.y + a.h - 4);

    for (const k of keys) {
      ctx.strokeStyle = "rgba(230,154,26,0.5)";
      ctx.beginPath(); ctx.moveTo(xOf(frameOf(k, nf)), a.y); ctx.lineTo(xOf(frameOf(k, nf)), a.y + a.h); ctx.stroke();
    }

    const curve = sampleCurve(keys, nf, interp());
    ctx.fillStyle = CURVE;
    for (let f = 0; f < nf; f++) {
      const v = curve[f];
      if (v === null) continue;
      ctx.fillRect(xOf(f) - 0.6, yOf(v) - 0.6, 1.4, 1.4);
    }

    keys.forEach((k, i) => {
      const x = xOf(frameOf(k, nf));
      const known = k.value !== null;
      const yv = yOf(known ? k.value : 0.5);
      const hot = i === st.hover || i === st.drag;
      ctx.beginPath();
      ctx.arc(x, yv, HANDLE_R, 0, Math.PI * 2);
      if (known) {
        ctx.fillStyle = hot ? POINT_HOT : POINT;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#101014";
        ctx.stroke();
      } else {
        ctx.fillStyle = BG;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = GHOST;
        ctx.stroke();
        ctx.fillStyle = GHOST;
        ctx.font = "8px monospace";
        ctx.textAlign = "center";
        ctx.fillText("?", x, yv + 3);
      }
    });

    const hint = st.drag >= 0 && keys[st.drag]?.value !== null
      ? `frame ${frameOf(keys[st.drag], nf)}  ·  plane ${keys[st.drag].value.toFixed(3)} (${Math.round(keys[st.drag].value * 255)}/255)`
      : `${keys.length} key${keys.length === 1 ? "" : "s"}  ·  ${interp()}`;
    ctx.fillStyle = "#8e919b";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.textAlign = "center";
    ctx.fillText(hint, a.x + a.w / 2, a.y + a.h - 16);
    ctx.restore();
  }

  function commit(node) {
    const before = kw.value;
    kw.value = serialize(st.keys);
    if (before !== kw.value && kw.callback) kw.callback(kw.value);
    node.setDirtyCanvas(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
  }

  // reserve a slot in the widget layout at the bottom of the node
  for (const w of node.widgets) if (w.name === "focus_curve_reserved") return;
  const holder = node.addWidget("curve", "focus_curve_reserved", null, null);
  holder.options = Object.assign({}, holder.options, { serialize: false });
  holder.computeSize = (width) => [width, H];
  holder.draw = function (ctx, nd, width, y, height) { draw(ctx, nd, width, y, height); };

  const local = (e, node) => {
    const p = app.canvas.convertEventToCanvasOffset(e);
    return [p[0] - node.pos[0], p[1] - node.pos[1]];
  };

  const hit = (mx, my) => {
    const a = st.area;
    if (!a) return -1;
    const nf = n();
    for (let i = 0; i < st.keys.length; i++) {
      const k = st.keys[i];
      const x = a.x + (frameOf(k, nf) / Math.max(1, nf - 1)) * a.w;
      const yv = a.y + (1 - (k.value === null ? 0.5 : k.value)) * a.h;
      if (Math.hypot(mx - x, my - yv) <= HIT_R) return i;
    }
    return -1;
  };

  node.onMouseDown = function (e) {
    const a = st.area;
    if (!a || e.button !== 0) return false;
    const [mx, my] = local(e, this);
    if (mx < a.x || mx > a.x + a.w || my < a.y || my > a.y + a.h) return false;

    const idx = hit(mx, my);
    if (idx >= 0) {
      st.drag = idx;
    } else {
      const nf = n();
      const f = Math.round(clamp01((mx - a.x) / a.w) * (nf - 1));
      const v = clamp01(1 - (my - a.y) / a.h);
      st.keys.push({ raw: String(f), target: v.toFixed(3), value: v });
      st.keys.sort((p, q) => frameOf(p, nf) - frameOf(q, nf));
      st.drag = st.keys.findIndex((k) => frameOf(k, nf) === f && Math.abs(k.value - v) < 1e-6);
      commit(this);
    }
    st.hover = st.drag;
    this.setDirtyCanvas(true, true);
    return true;
  };

  node.onMouseMove = function (e) {
    const a = st.area;
    if (!a) return false;
    const [mx, my] = local(e, this);
    const inArea = mx >= a.x && mx <= a.x + a.w && my >= a.y && my <= a.y + a.h;

    if (e.buttons !== 1) {
      if (inArea) {
        const h = hit(mx, my);
        if (h !== st.hover) { st.hover = h; this.setDirtyCanvas(true, true); }
      } else if (st.hover !== -1) {
        st.hover = -1;
        this.setDirtyCanvas(true, true);
      }
      return false;
    }
    if (st.drag < 0) return false;

    const nf = n();
    const k = st.keys[st.drag];
    k.raw = String(Math.round(clamp01((mx - a.x) / a.w) * (nf - 1)));
    k.value = clamp01(1 - (my - a.y) / a.h);
    k.target = k.value.toFixed(3);
    commit(this);
    return true;
  };

  node.onMouseUp = function () {
    if (st.drag >= 0) {
      st.drag = -1;
      this.setDirtyCanvas(true, true);
    }
  };

  // right-click on a point deletes it
  node.onMouseDownOriginal = node.onMouseDown;
  const baseDown = node.onMouseDown;
  node.onMouseDown = function (e) {
    if (e.button === 2) {
      const a = st.area;
      if (a) {
        const [mx, my] = local(e, this);
        const idx = hit(mx, my);
        if (idx >= 0 && st.keys.length > 1) {
          st.keys.splice(idx, 1);
          st.hover = -1;
          commit(this);
          return true;
        }
      }
      return false;
    }
    return baseDown.call(this, e);
  };

  // re-parse when the text or the combo changes
  for (const w of [kw, iw, fw].filter(Boolean)) {
    const orig = w.callback;
    w.callback = function (v, ...rest) {
      if (orig) orig.call(this, v, ...rest);
      if (w === kw) st.keys = parseKeys(kw.value);
      node.setDirtyCanvas(true, true);
    };
  }

  node.setSize([Math.max(node.size[0], 430), Math.max(node.size[1], 620)]);
}