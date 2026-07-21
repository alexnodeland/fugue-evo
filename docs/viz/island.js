// docs/viz/island.js — "Islands and Migration"
// A 2x2 grid of island panels over a real 2-D benchmark landscape: four
// populations (green) evolving independently on their own copy of the terrain,
// each island's best in coral. On migration generations, violet arrows flash
// between panels along the CURRENT topology and receiving panels glow — the
// gene exchange must visually READ. Below: a rolling strip chart of per-island
// best fitness (raw, LOWER IS BETTER) vs generation, migration generations
// ticked in violet. The story on the default multimodal landscape (rastrigin):
// islands plateau independently at different local optima; migration injects
// better genes and stuck islands drop.
//
// WASM-ONLY compute: every number comes from the real fugue-evo crate through
// crates/fugue-evo-wasm (ExploreIsland, explore_landscape_grid,
// explore_landscape_info). No algorithm math is re-implemented here. When the
// wasm pkg is absent the widget renders a notice instead of animating
// (root gets data-fugue-backend="wasm"|"none" for the automated tests).
// Self-contained IIFE; assumes fugue-viz.js + fugue-evo-wasm-loader.js loaded.
(function () {
  "use strict";
  if (typeof window === "undefined" || !window.FugueViz) return;
  var FV = window.FugueViz;

  var N_ISLANDS = 4;        // fixed 2x2 grid
  var ISLAND_POP = 16;      // individuals per island
  var MAXH = 120;           // rolling strip-chart window (generations)
  var FLASH_DUR = 0.6;      // migration flash lifetime (seconds)
  var HEATN = 96;           // landscape grid resolution (cells per side)
  var GAP = 12;             // px between island panels
  var PANEL_MAX = 150;      // px, target island panel side
  var HEIGHT = 484;         // total canvas CSS height
  var PACE_HZ = 5;          // generations per second
  var LINE_ALPHA = [0.35, 0.5, 0.7, 0.9]; // per-island strip-line alphas

  // Island i sits at grid cell POSMAP[i] (cells row-major: 0 TL, 1 TR, 2 BL,
  // 3 BR), so the ring i -> (i+1)%4 walks the perimeter of the square.
  var POSMAP = [0, 1, 3, 2];
  var RING_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0]];
  var STAR_EDGES = [[1, 0], [2, 0], [3, 0]]; // spokes to/from island 0 (hub)
  var DIAG_EDGES = [[0, 2], [1, 3]];         // the two diagonals (full, faint)

  // small hex/rgb parser (FugueViz keeps its own private; the heat ramp needs one)
  function toRgb(col) {
    col = (col || "").trim();
    var m = /^#?([0-9a-f]{6})$/i.exec(col);
    if (m) { var n = parseInt(m[1], 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
    var r = /rgba?\(([^)]+)\)/.exec(col);
    if (r) { var p = r[1].split(","); return [parseInt(p[0], 10), parseInt(p[1], 10), parseInt(p[2], 10)]; }
    return [88, 166, 255];
  }

  function fmtF(v) {
    if (v == null || !isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a !== 0 && (a < 0.01 || a >= 1e4)) return v.toExponential(2);
    return v.toFixed(3);
  }

  // Init defers on FV.wasmReady (fugue-evo-wasm-loader.js): resolves to the
  // wasm-bindgen module namespace, or null when the pkg is absent.
  FV.register("island-migration", function (root, FV) {
    var p = FV.wasmReady || Promise.resolve(null);
    p.then(function (W) {
      if (!W) {
        root.setAttribute("data-fugue-backend", "none");
        var n = document.createElement("div");
        n.className = "fv-pg-notice";
        n.textContent = "This figure runs the real fugue-evo crate compiled to WebAssembly — the wasm package isn't available in this build.";
        root.appendChild(n);
        return;
      }
      root.setAttribute("data-fugue-backend", "wasm");
      realInit(root, FV, W);
    });
  });

  function realInit(root, FV, W) {
    // ------------------------------------------------------------------ state
    var seed = parseInt(root.getAttribute("data-seed") || "11", 10);
    if (!isFinite(seed) || seed < 1) seed = 11;
    var interval = parseInt(root.getAttribute("data-interval") || "8", 10);
    if (!isFinite(interval)) interval = 8;
    if (interval < 2) interval = 2;
    if (interval > 20) interval = 20;
    var topo = "ring";

    // Landscape is fixed per instance (data-landscape); bad names fall back.
    var landscapeName = (root.getAttribute("data-landscape") || "rastrigin").toLowerCase();
    var info;
    try {
      info = JSON.parse(W.explore_landscape_info(landscapeName));
    } catch (e) {
      landscapeName = "rastrigin";
      info = JSON.parse(W.explore_landscape_info(landscapeName));
    }
    var LO = info.lo, HI = info.hi;         // each panel's domain is [lo,hi]^2
    var OPT = info.optimum || null;         // [x,y] | null — yellow reference
    var OPTF = info.optimum_fitness;        // raw optimum value (lower=better)

    var engine = null;
    var state = null;      // last parsed step(): {gen, migrated, islands, global_best}
    var hist = [];         // rolling [{gen, f:[4], mig}] — raw best per island
    var flashLife = 0;     // 1 -> 0 over FLASH_DUR after a migration step
    var flashTopo = topo;  // topology captured at the flash's migration step

    // ------------------------------------------------ landscape heat (precomputed)
    // Raw grid values from the crate, computed ONCE per landscape; the painted
    // offscreen canvas is reused (drawImage) by all four panels every frame and
    // repainted only on theme change. Nonlinear ramp (sqrt of the normalized
    // goodness) so the basins stay visible against the peaks.
    var gridVals = W.explore_landscape_grid(landscapeName, LO, HI, LO, HI, HEATN, HEATN);
    var gmin = Infinity, gmax = -Infinity, gi;
    for (gi = 0; gi < gridVals.length; gi++) {
      var gv = gridVals[gi];
      if (!isFinite(gv)) continue;
      if (gv < gmin) gmin = gv;
      if (gv > gmax) gmax = gv;
    }
    if (!isFinite(gmin) || !(gmax > gmin)) { gmin = 0; gmax = 1; }
    var heatCanvas = document.createElement("canvas");
    var heatCtx = heatCanvas.getContext("2d");
    function paintHeat() {
      heatCanvas.width = HEATN;
      heatCanvas.height = HEATN;
      heatCtx.clearRect(0, 0, HEATN, HEATN);
      var rgb = toRgb(FV.theme().colors.prior);
      var span = gmax - gmin;
      for (var j = 0; j < HEATN; j++) {
        for (var i = 0; i < HEATN; i++) {
          var v = gridVals[j * HEATN + i]; // row-major, j indexes y from ymin
          var norm = isFinite(v) ? (v - gmin) / span : 1;
          if (norm < 0) norm = 0;
          if (norm > 1) norm = 1;
          var a = Math.sqrt(1 - norm) * 0.85; // lower raw value = brighter basin
          if (a <= 0.02) continue;
          heatCtx.fillStyle = "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + "," + a + ")";
          // canvas y grows downward; grid j grows from ymin — flip.
          heatCtx.fillRect(i, HEATN - 1 - j, 1, 1);
        }
      }
    }
    paintHeat();

    // ----------------------------------------------------------------- engine
    function rebuild() {
      try {
        engine = new W.ExploreIsland(landscapeName, N_ISLANDS, ISLAND_POP, interval, topo, BigInt(seed));
      } catch (e) {
        engine = null;
        try { console.warn("[fugue-viz] island-migration: engine build failed", e); } catch (e2) { /* readout only */ }
        return;
      }
      state = null;
      hist = [];
      flashLife = 0;
      flashTopo = topo;
      // Reduced motion: step ~30 generations synchronously and draw once (a
      // fully-formed static frame — no flashes, no animation). Otherwise a
      // couple of warm-up steps so panels and chart aren't empty at first paint.
      var warm = (loopApi && loopApi.reduced) ? 30 : 2;
      for (var i = 0; i < warm; i++) stepOnce();
      flashLife = 0;
      renderReadouts();
      draw();
    }

    function stepOnce() {
      if (!engine) return;
      var s;
      try {
        s = JSON.parse(engine.step()); // step() returns a JSON STRING
      } catch (e) {
        try { console.warn("[fugue-viz] island-migration: step failed", e); } catch (e2) { /* readout only */ }
        if (loopApi) loopApi.pause();
        return;
      }
      state = s;
      var fs = new Array(N_ISLANDS);
      for (var i = 0; i < N_ISLANDS; i++) {
        var isl = s.islands[i];
        fs[i] = (isl && isl.best) ? isl.best[2] : NaN; // raw, lower is better
      }
      hist.push({ gen: s.gen, f: fs, mig: !!s.migrated });
      if (hist.length > MAXH) hist.shift();
      if (s.migrated) {
        flashTopo = topo;
        if (!(loopApi && loopApi.reduced)) flashLife = 1;
      }
      renderReadouts();
    }

    // --------------------------------------------------------------- DOM shell
    var controls = document.createElement("div");
    controls.className = "fv-controls";
    root.appendChild(controls);

    FV.slider(controls, {
      label: "MIGRATE EVERY", min: 2, max: 20, step: 1, value: interval,
      fmt: function (v) { return (v | 0) + " gen"; },
      onInput: function (v) { interval = v | 0; rebuild(); }
    });

    var topoBtns = FV.buttons(controls, [
      { label: "Ring", title: "Ring topology: island i sends to island i+1", onClick: function () { setTopo("ring"); } },
      { label: "Star", title: "Star topology: spokes to/from island 0", onClick: function () { setTopo("star"); } },
      { label: "Full", title: "Fully connected: every island exchanges with every other", onClick: function () { setTopo("full"); } }
    ]);
    var TOPO_LABEL = { ring: "Ring", star: "Star", full: "Full" };
    function styleTopoButtons() {
      for (var k in TOPO_LABEL) {
        if (!TOPO_LABEL.hasOwnProperty(k)) continue;
        var b = topoBtns.fvButtons[TOPO_LABEL[k]];
        if (!b) continue;
        if (k === topo) b.classList.add("fv-primary");
        else b.classList.remove("fv-primary");
      }
    }
    function setTopo(t) {
      if (t === topo) return;
      topo = t;
      styleTopoButtons();
      rebuild();
    }
    styleTopoButtons();

    var btns = FV.buttons(controls, [
      { label: "Play", title: "Run the islands", primary: true, onClick: function () { togglePlay(); } },
      { label: "Reset", title: "Rebuild from the current seed", onClick: function () { rebuild(); } }
    ]);

    // Seed: always visible, Victor-scrubbable; changing it rebuilds the engine
    // deterministically (u64 crosses wasm-bindgen as BigInt).
    var seedCtl = document.createElement("label");
    seedCtl.className = "fv-control";
    var seedLab = document.createElement("span");
    seedLab.className = "fv-control-label";
    seedLab.textContent = "SEED";
    seedCtl.appendChild(seedLab);
    var seedSpan = document.createElement("span");
    seedCtl.appendChild(seedSpan);
    controls.appendChild(seedCtl);
    FV.scrub(seedSpan, {
      min: 1, max: 9999, step: 1, value: seed,
      onInput: function (v) { seed = v | 0; rebuild(); }
    });

    var cv = FV.canvas(root, { height: HEIGHT, onResize: function () { draw(); } });
    var ctx = cv.ctx;

    var instr = document.createElement("div");
    instr.className = "fv-instruction";
    instr.textContent = "blue terrain = the real landscape (bright = basin) · green = island populations · coral = island best · violet flash = migration";
    root.appendChild(instr);

    var readouts = document.createElement("div");
    readouts.className = "fv-readouts";
    root.appendChild(readouts);
    var rGen = FV.readout(readouts, { label: "GENERATION" });
    var rBest = FV.readout(readouts, { label: "GLOBAL BEST f" });

    var hint = document.createElement("div");
    hint.className = "fv-hint";
    hint.textContent = "watch each island plateau on its own local optimum, then a violet migration inject better genes — stuck islands drop in the chart. Compare ring vs star vs full, and stretch the migration interval.";
    root.appendChild(hint);

    function renderReadouts() {
      rGen.set(state ? String(state.gen) : "—");
      var gb = state && state.global_best ? state.global_best[2] : null;
      rBest.set(fmtF(gb), "hot");
    }

    // ----------------------------------------------------------------- layout
    var panels = null;  // [{x, y, s, sx, sy, cx, cy}] indexed by island
    var strip = null;   // {ix, iy, iw, ih}
    function layout(w) {
      var s = Math.min(PANEL_MAX, Math.floor((w - GAP) / 2));
      if (s < 60) s = 60;
      var gridW = 2 * s + GAP;
      var gx = Math.max(0, (w - gridW) / 2);
      var gy = 8;
      panels = new Array(N_ISLANDS);
      for (var i = 0; i < N_ISLANDS; i++) {
        var cell = POSMAP[i];
        var col = cell % 2, row = (cell / 2) | 0;
        var x = gx + col * (s + GAP);
        var y = gy + row * (s + GAP);
        panels[i] = {
          x: x, y: y, s: s,
          sx: FV.scale([LO, HI], [x, x + s]),
          sy: FV.scale([LO, HI], [y + s, y]),
          cx: x + s / 2, cy: y + s / 2
        };
      }
      var stripY = gy + 2 * s + GAP + 18;
      var padL = 46, padR = 10, padB = 30;
      strip = {
        ix: padL, iy: stripY,
        iw: Math.max(40, w - padL - padR),
        ih: Math.max(60, HEIGHT - stripY - padB)
      };
    }

    // ------------------------------------------------------------------- draw
    function drawArrow(x0, y0, x1, y1, color, alpha, width) {
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width || 2.5;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      var ang = Math.atan2(y1 - y0, x1 - x0), hl = 8;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - hl * Math.cos(ang - 0.4), y1 - hl * Math.sin(ang - 0.4));
      ctx.lineTo(x1 - hl * Math.cos(ang + 0.4), y1 - hl * Math.sin(ang + 0.4));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function drawPanel(i, th, C) {
      var r = panels[i];
      // precomputed landscape heat, scaled into this panel
      ctx.drawImage(heatCanvas, r.x, r.y, r.s, r.s);
      // frame
      ctx.save();
      ctx.strokeStyle = C.ink;
      ctx.globalAlpha = 0.3;
      ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.s - 1, r.s - 1);
      ctx.restore();

      // dots CLIP to the panel's frame
      ctx.save();
      ctx.beginPath();
      ctx.rect(r.x, r.y, r.s, r.s);
      ctx.clip();

      // yellow reference: the landscape's known optimum
      if (OPT) {
        var ox = r.sx(OPT[0]), oy = r.sy(OPT[1]);
        ctx.save();
        ctx.strokeStyle = C.data;
        ctx.globalAlpha = 0.75;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ox - 4, oy); ctx.lineTo(ox + 4, oy);
        ctx.moveTo(ox, oy - 4); ctx.lineTo(ox, oy + 4);
        ctx.stroke();
        ctx.restore();
      }

      var isl = state && state.islands ? state.islands[i] : null;
      if (isl) {
        // population — green
        ctx.save();
        ctx.fillStyle = C.post;
        ctx.globalAlpha = 0.85;
        for (var k = 0; k < isl.population.length; k++) {
          var ind = isl.population[k];
          ctx.beginPath();
          ctx.arc(r.sx(ind[0]), r.sy(ind[1]), 2, 0, 2 * Math.PI);
          ctx.fill();
        }
        ctx.restore();
        // island best — coral with a glow
        if (isl.best) {
          var bx = r.sx(isl.best[0]), by = r.sy(isl.best[1]);
          ctx.save();
          ctx.fillStyle = C.hot;
          ctx.globalAlpha = 0.28;
          ctx.beginPath(); ctx.arc(bx, by, 7, 0, 2 * Math.PI); ctx.fill();
          ctx.globalAlpha = 1;
          ctx.beginPath(); ctx.arc(bx, by, 3, 0, 2 * Math.PI); ctx.fill();
          ctx.restore();
        }
      }
      ctx.restore(); // end panel clip

      // label
      ctx.save();
      ctx.fillStyle = C.ink;
      ctx.globalAlpha = 0.65;
      ctx.font = "10px var(--mono-font, monospace)";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("island " + i, r.x + 5, r.y + 4);
      ctx.restore();

      // brief flow halo on receiving islands while a migration flash is live
      // (every island receives under ring / star hub-exchange / full)
      if (flashLife > 0) {
        ctx.save();
        ctx.strokeStyle = C.flow;
        ctx.globalAlpha = 0.85 * flashLife;
        ctx.lineWidth = 3;
        ctx.strokeRect(r.x - 2, r.y - 2, r.s + 4, r.s + 4);
        ctx.restore();
      }
    }

    function edgePoints(a, b) {
      // arrow endpoints pulled in from the panel centers so heads stay legible
      var pa = panels[a], pb = panels[b];
      var dx = pb.cx - pa.cx, dy = pb.cy - pa.cy;
      var len = Math.sqrt(dx * dx + dy * dy) || 1;
      var inset = pa.s * 0.2;
      return [
        pa.cx + (dx / len) * inset, pa.cy + (dy / len) * inset,
        pb.cx - (dx / len) * inset, pb.cy - (dy / len) * inset
      ];
    }

    function drawMigration(C) {
      if (flashLife <= 0) return;
      var a = flashLife;
      var e, pts, i;
      if (flashTopo === "ring") {
        for (i = 0; i < RING_EDGES.length; i++) {
          e = RING_EDGES[i];
          pts = edgePoints(e[0], e[1]);
          drawArrow(pts[0], pts[1], pts[2], pts[3], C.flow, 0.9 * a, 2.5);
        }
      } else if (flashTopo === "star") {
        // spokes: leaves -> hub and hub -> leaves (exchange both ways)
        for (i = 0; i < STAR_EDGES.length; i++) {
          e = STAR_EDGES[i];
          pts = edgePoints(e[0], e[1]);
          drawArrow(pts[0], pts[1], pts[2], pts[3], C.flow, 0.9 * a, 2.5);
          drawArrow(pts[2], pts[3], pts[0], pts[1], C.flow, 0.55 * a, 2);
        }
      } else {
        // full: the 4-cycle plus the diagonals, diagonals faint
        for (i = 0; i < RING_EDGES.length; i++) {
          e = RING_EDGES[i];
          pts = edgePoints(e[0], e[1]);
          drawArrow(pts[0], pts[1], pts[2], pts[3], C.flow, 0.9 * a, 2.5);
        }
        for (i = 0; i < DIAG_EDGES.length; i++) {
          e = DIAG_EDGES[i];
          pts = edgePoints(e[0], e[1]);
          drawArrow(pts[0], pts[1], pts[2], pts[3], C.flow, 0.35 * a, 2);
          drawArrow(pts[2], pts[3], pts[0], pts[1], C.flow, 0.35 * a, 2);
        }
      }
    }

    function drawStrip(th, C) {
      var st = strip;
      var n = hist.length;
      var g0 = n ? hist[0].gen : 0;
      var g1 = n ? hist[n - 1].gen : 0;
      if (g1 - g0 < 10) g1 = g0 + 10;
      var xs = FV.scale([g0, g1], [st.ix, st.ix + st.iw]);

      // y domain over the window's finite bests, anchored to the optimum so
      // "distance left to the optimum" always reads. Raw values, LOWER = BETTER.
      var ylo = Infinity, yhi = -Infinity, i, k, v;
      for (i = 0; i < n; i++) {
        for (k = 0; k < N_ISLANDS; k++) {
          v = hist[i].f[k];
          if (!isFinite(v)) continue;
          if (v < ylo) ylo = v;
          if (v > yhi) yhi = v;
        }
      }
      if (isFinite(OPTF)) {
        if (OPTF < ylo) ylo = OPTF;
        if (OPTF > yhi) yhi = OPTF;
      }
      if (!isFinite(ylo) || !isFinite(yhi)) { ylo = 0; yhi = 1; }
      var pad = (yhi - ylo || 1) * 0.08;
      var ys = FV.scale([ylo - pad, yhi + pad], [st.iy + st.ih, st.iy]);

      FV.axes(ctx, {
        x: st.ix, y: st.iy, w: st.iw, h: st.ih,
        xscale: xs, yscale: ys,
        xlabel: "generation", ylabel: "best f", theme: th
      });

      ctx.save();
      ctx.beginPath();
      ctx.rect(st.ix, st.iy, st.iw, st.ih);
      ctx.clip();

      // migration generations — vertical flow ticks
      ctx.save();
      ctx.strokeStyle = C.flow;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 1;
      for (i = 0; i < n; i++) {
        if (!hist[i].mig) continue;
        var mx = xs(hist[i].gen);
        ctx.beginPath();
        ctx.moveTo(mx, st.iy);
        ctx.lineTo(mx, st.iy + st.ih);
        ctx.stroke();
      }
      ctx.restore();

      // the optimum — yellow dashed reference
      if (isFinite(OPTF)) {
        ctx.save();
        ctx.globalAlpha = 0.55;
        FV.curve(ctx, [[st.ix, ys(OPTF)], [st.ix + st.iw, ys(OPTF)]], { color: C.data, width: 1, dash: [4, 3] });
        ctx.restore();
      }

      // four thin per-island best lines, green at staggered alphas
      for (k = 0; k < N_ISLANDS; k++) {
        var pts = new Array(n);
        for (i = 0; i < n; i++) {
          v = hist[i].f[k];
          pts[i] = isFinite(v) ? [xs(hist[i].gen), ys(v)] : null; // FV.curve skips nulls
        }
        ctx.save();
        ctx.globalAlpha = LINE_ALPHA[k];
        FV.curve(ctx, pts, { color: C.post, width: 1.3 });
        ctx.restore();
      }

      ctx.restore(); // end strip clip
    }

    function draw() {
      if (!cv) return; // onResize can fire during canvas() before cv is assigned
      cv.clear();
      var th = FV.theme(), C = th.colors;
      layout(cv.w);
      for (var i = 0; i < N_ISLANDS; i++) drawPanel(i, th, C);
      drawMigration(C);
      drawStrip(th, C);
    }

    // ------------------------------------------------------------------- loop
    var pacer = FV.pace(PACE_HZ);
    var loopApi = FV.loop(root, function (dt) {
      if (dt === 0) {
        stepOnce();
      } else {
        for (var n = pacer(dt); n-- > 0;) stepOnce();
        // Flash decays on wall-clock time so it lasts ~FLASH_DUR at any fps.
        if (flashLife > 0) {
          flashLife -= dt / FLASH_DUR;
          if (flashLife < 0) flashLife = 0;
        }
      }
      draw();
    }, { autoplay: true });

    function togglePlay() {
      if (loopApi.playing) {
        loopApi.pause();
        btns.fvButtons["Play"].textContent = "Play";
      } else {
        loopApi.play();
        if (loopApi.playing) btns.fvButtons["Play"].textContent = "Pause";
      }
    }

    if (loopApi.reduced) {
      btns.fvButtons["Play"].disabled = true;
      btns.fvButtons["Play"].title = "Reduced motion is on — the figure shows a converged static frame";
    }

    FV.onThemeChange(function () { paintHeat(); draw(); });

    // first paint: build the engine (rebuild warms up ~2 gens, or ~30 under
    // reduced motion so the static frame already tells the story) and draw.
    rebuild();
    if (loopApi.playing) btns.fvButtons["Play"].textContent = "Pause";
  }
})();
