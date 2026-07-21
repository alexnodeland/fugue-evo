// docs/viz/cmaes.js — "The Adapting Ellipse" (CMA-ES over a real landscape)
//
// DATA-FIRST: a live CMA-ES population walking a real benchmark landscape.
// Single wide canvas: landscape heatmap (data/yellow, basins bright), the
// population of the current generation (post/green dots), the search mean
// (hot/coral dot) with a fading trail of past means, and the 1σ / 2σ
// covariance ellipses of the sampling distribution (prior/blue; 2σ dashed,
// fainter) — watch the ellipse elongate along rosenbrock's valley and shrink
// to a point at convergence. Best-so-far wears a yellow ring.
//
// WASM-ONLY compute: every number comes from the real fugue-evo crate via
// crates/fugue-evo-wasm (ExploreCma / explore_landscape_grid /
// explore_landscape_info). No algorithm math is re-implemented here — if the
// wasm pkg is absent the widget shows a notice instead of animating
// (root gets data-fugue-backend="wasm" | "none"; automated tests read it).
//
// data-* options: data-seed (default 11), data-landscape (default
// "rosenbrock"), data-sigma (default 1.5).
// Self-contained IIFE; assumes fugue-viz.js + fugue-evo-wasm-loader.js loaded.
(function () {
  "use strict";
  if (typeof window === "undefined" || !window.FugueViz) return;
  var FV = window.FugueViz;

  var LANDSCAPES = ["sphere", "rastrigin", "rosenbrock", "ackley"];
  var TRAIL_MAX = 60;      // retained past means (fading coral trail)
  var PACE_HZ = 3;         // generations per second while playing
  var REDUCED_GENS = 25;   // synchronous generations under reduced motion
  var HEAT_STEP = 4;       // heatmap cell size in CSS px

  function toRgb(col) {
    col = (col || "").trim();
    var m = /^#?([0-9a-f]{6})$/i.exec(col);
    if (m) { var n = parseInt(m[1], 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
    var r = /rgba?\(([^)]+)\)/.exec(col);
    if (r) { var p = r[1].split(","); return [parseInt(p[0], 10), parseInt(p[1], 10), parseInt(p[2], 10)]; }
    return [242, 204, 96];
  }

  function fmtF(v) {
    if (!isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(2);
    return Number(v.toPrecision(3)).toString();
  }

  // Init defers on FV.wasmReady: the wasm-bindgen module namespace, or null
  // when the pkg is absent — then the widget declines to imitate the crate.
  FV.register("cmaes-ellipse", function (root, FV) {
    var p = FV.wasmReady || Promise.resolve(null);
    p.then(function (mod) {
      if (!mod) {
        root.setAttribute("data-fugue-backend", "none");
        var note = document.createElement("div");
        note.className = "fv-pg-notice";
        note.textContent = "This figure runs the real fugue-evo crate compiled to WebAssembly — the wasm package isn't available in this build.";
        root.appendChild(note);
        return;
      }
      root.setAttribute("data-fugue-backend", "wasm");
      realInit(root, FV, mod);
    });
  });

  function realInit(root, FV, W) {
    // ------------------------------------------------------------------ state
    var params = {
      seed: parseInt(root.getAttribute("data-seed") || "11", 10) || 11,
      landscape: (root.getAttribute("data-landscape") || "rosenbrock").toLowerCase(),
      sigma0: parseFloat(root.getAttribute("data-sigma") || "1.5") || 1.5
    };
    if (LANDSCAPES.indexOf(params.landscape) < 0) params.landscape = "rosenbrock";
    if (params.sigma0 < 0.1) params.sigma0 = 0.1;
    if (params.sigma0 > 3.0) params.sigma0 = 3.0;

    var engine = null;     // W.ExploreCma
    var state = null;      // last JSON.parse'd step()
    var trail = [];        // past means [[x,y],...], capped at TRAIL_MAX
    var wasConverged = false;
    var info = null;       // {lo, hi, optimum, ...} for the active landscape
    var plot = null;       // layout: {ix, iy, iw, ih, sx, sy}

    function loadInfo() {
      info = JSON.parse(W.explore_landscape_info(params.landscape));
    }

    // One real CMA-ES generation via wasm; returns false when stepping failed
    // (e.g. degenerate covariance long after convergence) — the caller pauses.
    function stepOnce() {
      if (!engine) return false;
      var s;
      try {
        s = JSON.parse(engine.step());
      } catch (e) {
        try { console.warn("[fugue-viz] cmaes-ellipse: step failed", e); } catch (e2) { /* readout only */ }
        return false;
      }
      state = s;
      trail.push([s.mean[0], s.mean[1]]);
      if (trail.length > TRAIL_MAX) trail.shift();
      return true;
    }

    // Rebuild the engine deterministically from (landscape, σ₀, seed). Start
    // at (0.7·hi, 0.7·lo) — an off-center corner so there is a visible journey.
    function rebuild() {
      loadInfo();
      var x0 = 0.7 * info.hi;
      var y0 = 0.7 * info.lo;
      // Seeds are u64 at the wasm boundary — BigInt. lambda 0 = library default.
      engine = new W.ExploreCma(params.landscape, x0, y0, params.sigma0, 0, BigInt(params.seed));
      state = null;
      trail = [];
      wasConverged = false;
      heatDirty = true;
      // Always land on a meaningful frame: reduced motion gets a ~25-gen
      // synchronous run (its only frame), everyone else gets generation 1.
      var warm = loopApi && loopApi.reduced ? REDUCED_GENS : 1;
      for (var i = 0; i < warm; i++) {
        if (!stepOnce()) break;
        if (state && state.converged) break;
      }
      renderReadouts();
    }

    // --------------------------------------------------------------- DOM shell
    var controls = document.createElement("div");
    controls.className = "fv-controls";
    root.appendChild(controls);

    // landscape select
    var landCtl = document.createElement("label");
    landCtl.className = "fv-control";
    var landLab = document.createElement("span");
    landLab.className = "fv-control-label";
    landLab.textContent = "LANDSCAPE";
    landCtl.appendChild(landLab);
    var landSel = document.createElement("select");
    landSel.className = "fv-select";
    for (var li = 0; li < LANDSCAPES.length; li++) {
      var opt = document.createElement("option");
      opt.value = LANDSCAPES[li];
      opt.textContent = LANDSCAPES[li];
      landSel.appendChild(opt);
    }
    landSel.value = params.landscape;
    landSel.addEventListener("change", function () {
      params.landscape = landSel.value;
      rebuild();
      requestDraw();
    });
    landCtl.appendChild(landSel);
    controls.appendChild(landCtl);

    // σ₀ scrub (Victor-style draggable number)
    var sigCtl = document.createElement("div");
    sigCtl.className = "fv-control";
    var sigLab = document.createElement("span");
    sigLab.className = "fv-control-label";
    sigLab.textContent = "INITIAL σ₀";
    sigCtl.appendChild(sigLab);
    var sigSpan = document.createElement("span");
    sigCtl.appendChild(sigSpan);
    controls.appendChild(sigCtl);
    FV.scrub(sigSpan, {
      min: 0.1, max: 3.0, step: 0.1, value: params.sigma0,
      fmt: function (v) { return v.toFixed(1); },
      onInput: function (v) {
        params.sigma0 = v;
        rebuild();
        requestDraw();
      }
    });

    // seed scrub — always visible; changing it replays a different recording
    var seedCtl = document.createElement("div");
    seedCtl.className = "fv-control";
    var seedLab = document.createElement("span");
    seedLab.className = "fv-control-label";
    seedLab.textContent = "SEED";
    seedCtl.appendChild(seedLab);
    var seedSpan = document.createElement("span");
    seedCtl.appendChild(seedSpan);
    controls.appendChild(seedCtl);
    FV.scrub(seedSpan, {
      min: 1, max: 999, step: 1, value: params.seed,
      fmt: function (v) { return String(v | 0); },
      onInput: function (v) {
        params.seed = v | 0;
        rebuild();
        requestDraw();
      }
    });

    var btns = FV.buttons(controls, [
      { label: "Play", title: "Run generations (~3/s)", primary: true, onClick: function () { togglePlay(); } },
      { label: "Step", title: "One generation", onClick: function () { loopApi.step(); } },
      { label: "Reset", title: "Restart from (0.7·hi, 0.7·lo) with the same seed", onClick: function () { rebuild(); requestDraw(); } }
    ]);

    var cv = FV.canvas(root, { height: 340, onResize: function () { heatDirty = true; draw(); } });
    var ctx = cv.ctx;

    var instr = document.createElement("div");
    instr.className = "fv-instruction";
    instr.textContent = "blue ellipses = 1σ / 2σ of the sampling distribution · green = this generation's samples · coral = mean and its path · yellow ring = best so far";
    root.appendChild(instr);

    var readouts = document.createElement("div");
    readouts.className = "fv-readouts";
    root.appendChild(readouts);
    var rGen = FV.readout(readouts, { label: "GEN" });
    var rSigma = FV.readout(readouts, { label: "σ (STEP SIZE)" });
    var rBest = FV.readout(readouts, { label: "BEST f" });

    var hint = document.createElement("div");
    hint.className = "fv-hint";
    hint.textContent = "on rosenbrock, watch the ellipse elongate along the curved valley, then shrink to a dot as σ collapses at the optimum — scrub σ₀ down to 0.1 and the search crawls instead.";
    root.appendChild(hint);

    function renderReadouts() {
      if (!state) { rGen.set("—"); rSigma.set("—"); rBest.set("—"); return; }
      rGen.set(String(state.gen));
      // σ goes green once the run has converged (the ellipse is done adapting).
      rSigma.set(fmtF(state.sigma), state.converged ? "post" : null);
      rBest.set(fmtF(state.best[2]));
    }

    // ---------------------------------------------------------------- heatmap
    // Offscreen landscape heatmap in the data (yellow) role. Raw values come
    // from the crate (lower = better); normalized with a LOG ramp and
    // INVERTED so basins glow — rosenbrock spans six orders of magnitude and
    // anything gentler flattens its valley to an even wash. Painted per-pixel
    // into a grid-resolution ImageData, then drawImage-scaled (smoothed), so
    // there are no per-cell seams.
    var heatDirty = true;
    var heatCanvas = document.createElement("canvas");
    var heatCtx = heatCanvas.getContext("2d");
    function rebuildHeat(iw, ih, col) {
      iw = Math.max(1, Math.round(iw)); ih = Math.max(1, Math.round(ih));
      heatCanvas.width = iw; heatCanvas.height = ih;
      var nx = Math.max(2, Math.ceil(iw / HEAT_STEP));
      var ny = Math.max(2, Math.ceil(ih / HEAT_STEP));
      var vals = W.explore_landscape_grid(params.landscape, info.lo, info.hi, info.lo, info.hi, nx, ny);
      var minv = Infinity, maxv = -Infinity, k;
      for (k = 0; k < vals.length; k++) {
        var v = vals[k];
        if (!isFinite(v)) continue;
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
      }
      var span = maxv - minv;
      if (!isFinite(span) || span <= 0) span = 1;
      var logSpan = Math.log(1 + span);
      var rgb = toRgb(col);
      var cell = document.createElement("canvas");
      cell.width = nx; cell.height = ny;
      var cellCtx = cell.getContext("2d");
      var img = cellCtx.createImageData(nx, ny);
      for (var j = 0; j < ny; j++) {
        for (var i = 0; i < nx; i++) {
          // grid is row-major with j indexing y from info.lo up; canvas rows
          // grow downward, so row j paints at (ny - 1 - j).
          var raw = vals[j * nx + i];
          var t = isFinite(raw) ? Math.log(1 + (raw - minv)) / logSpan : 1;
          var a = (1 - t) * 0.55;     // basins (low raw value) bright
          var p = (((ny - 1 - j) * nx) + i) * 4;
          img.data[p] = rgb[0];
          img.data[p + 1] = rgb[1];
          img.data[p + 2] = rgb[2];
          img.data[p + 3] = Math.max(0, Math.round(a * 255));
        }
      }
      cellCtx.putImageData(img, 0, 0);
      heatCtx.clearRect(0, 0, iw, ih);
      heatCtx.imageSmoothingEnabled = true;
      heatCtx.drawImage(cell, 0, 0, iw, ih);
      heatDirty = false;
    }

    // ----------------------------------------------------------------- layout
    function layout(w, h) {
      var pad = { l: 40, r: 12, t: 10, b: 26 };
      var iw = Math.max(10, w - pad.l - pad.r);
      var ih = Math.max(10, h - pad.t - pad.b);
      plot = {
        ix: pad.l, iy: pad.t, iw: iw, ih: ih,
        sx: FV.scale([info.lo, info.hi], [pad.l, pad.l + iw]),
        sy: FV.scale([info.lo, info.hi], [pad.t + ih, pad.t])
      };
    }

    // Ellipse of the SAMPLING covariance σ²·C as a polyline in data coords —
    // exact under the (non-uniform) x/y pixel scales, and clipped like any
    // other curve. Angle derived from cov directly (do not trust the wasm
    // eigenvector orientation convention): θ = ½·atan2(2c01, c00 − c11),
    // axis radii k·σ·√λᵢ with λᵢ the eigenvalues of C.
    function ellipsePts(mx, my, r1, r2, theta) {
      var n = 64, pts = new Array(n + 1);
      var ct = Math.cos(theta), st = Math.sin(theta);
      for (var i = 0; i <= n; i++) {
        var t = (i / n) * 2 * Math.PI;
        var ex = r1 * Math.cos(t), ey = r2 * Math.sin(t);
        pts[i] = [plot.sx(mx + ex * ct - ey * st), plot.sy(my + ex * st + ey * ct)];
      }
      return pts;
    }

    function covEllipse(s, k) {
      var c00 = s.cov[0], c01 = (s.cov[1] + s.cov[2]) / 2, c11 = s.cov[3];
      var half = (c00 + c11) / 2;
      var d = Math.sqrt(Math.max(0, ((c00 - c11) / 2) * ((c00 - c11) / 2) + c01 * c01));
      var l1 = Math.max(0, half + d);   // eigenvalues of C, l1 >= l2
      var l2 = Math.max(0, half - d);
      var theta = 0.5 * Math.atan2(2 * c01, c00 - c11);   // major-axis angle
      return ellipsePts(s.mean[0], s.mean[1], k * s.sigma * Math.sqrt(l1), k * s.sigma * Math.sqrt(l2), theta);
    }

    // ------------------------------------------------------------------- draw
    function dotRaw(x, y, r, color) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); }
    function dot(x, y, r, color, alpha) { ctx.save(); ctx.globalAlpha = alpha; dotRaw(x, y, r, color); ctx.restore(); }

    function draw() {
      if (!cv || !info) return; // onResize can fire during canvas() setup
      var w = cv.w, h = cv.h;
      cv.clear();
      var th = FV.theme(), C = th.colors;
      layout(w, h);
      var pl = plot;

      // landscape heatmap (offscreen, rebuilt only when dirty/resized)
      if (heatDirty || heatCanvas.width !== Math.round(pl.iw) || heatCanvas.height !== Math.round(pl.ih)) {
        rebuildHeat(pl.iw, pl.ih, C.data);
      }
      ctx.drawImage(heatCanvas, pl.ix, pl.iy, pl.iw, pl.ih);
      FV.axes(ctx, { x: pl.ix, y: pl.iy, w: pl.iw, h: pl.ih, xscale: pl.sx, yscale: pl.sy, xlabel: "x", ylabel: "y", theme: th });

      // the known optimum, if the landscape has one — faint yellow cross
      if (state && info.optimum) {
        var ox = pl.sx(info.optimum[0]), oy = pl.sy(info.optimum[1]);
        ctx.save();
        ctx.strokeStyle = C.data; ctx.globalAlpha = 0.7; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ox - 5, oy); ctx.lineTo(ox + 5, oy);
        ctx.moveTo(ox, oy - 5); ctx.lineTo(ox, oy + 5);
        ctx.stroke();
        ctx.restore();
      }

      if (!state) return;

      // Everything dynamic clips to the frame: real CMA-ES means, samples and
      // ellipses can wander outside the bounds on their way in.
      ctx.save();
      ctx.beginPath();
      ctx.rect(pl.ix, pl.iy, pl.iw, pl.ih);
      ctx.clip();

      // fading trail of past means — coral, per-segment decaying alpha
      // (newest segment brightest, oldest nearly gone; capped at TRAIL_MAX)
      var N = trail.length, i;
      ctx.save();
      ctx.strokeStyle = C.hot;
      ctx.lineWidth = 1.4;
      ctx.lineCap = "round";
      for (i = 1; i < N; i++) {
        ctx.globalAlpha = 0.06 + 0.55 * (i / N);
        ctx.beginPath();
        ctx.moveTo(pl.sx(trail[i - 1][0]), pl.sy(trail[i - 1][1]));
        ctx.lineTo(pl.sx(trail[i][0]), pl.sy(trail[i][1]));
        ctx.stroke();
      }
      ctx.restore();

      // covariance ellipses of the sampling distribution σ²·C (prior/blue):
      // 2σ dashed + fainter under the solid 1σ
      ctx.save();
      ctx.globalAlpha = 0.45;
      FV.curve(ctx, covEllipse(state, 2), { color: C.prior, width: 1.2, dash: [5, 4] });
      ctx.restore();
      ctx.save();
      ctx.globalAlpha = 0.9;
      FV.curve(ctx, covEllipse(state, 1), { color: C.prior, width: 1.8 });
      ctx.restore();

      // this generation's population — green
      var pop = state.population || [];
      for (i = 0; i < pop.length; i++) {
        dot(pl.sx(pop[i][0]), pl.sy(pop[i][1]), 3, C.post, 0.85);
      }

      // best-so-far — yellow ring
      ctx.save();
      ctx.strokeStyle = C.data;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.95;
      ctx.beginPath();
      ctx.arc(pl.sx(state.best[0]), pl.sy(state.best[1]), 7, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.restore();

      // the search mean — coral with a glow
      var mx = pl.sx(state.mean[0]), my = pl.sy(state.mean[1]);
      ctx.save(); ctx.globalAlpha = 0.25; dotRaw(mx, my, 9, C.hot); ctx.restore();
      dot(mx, my, 4.5, C.hot, 1);

      ctx.restore(); // end clip
    }

    // schedule a single draw when paused (the loop draws while playing)
    var drawQueued = false;
    function requestDraw() {
      if (loopApi && loopApi.playing) return;
      if (drawQueued) return;
      drawQueued = true;
      window.requestAnimationFrame(function () { drawQueued = false; draw(); });
    }

    // ------------------------------------------------------------------- loop
    // ~3 generations/sec via FV.pace so the covariance adaptation is watchable;
    // Step (dt === 0) advances exactly one generation.
    var pacer = FV.pace(PACE_HZ);
    var loopApi = FV.loop(root, function (dt) {
      var n = dt === 0 ? 1 : pacer(dt);
      var moved = false;
      for (var i = 0; i < n; i++) {
        if (!stepOnce()) { loopApi.pause(); setPlayLabel(); break; }
        moved = true;
        if (state.converged) break;
      }
      if (moved) renderReadouts();
      // Pause once on the transition into convergence — σ readout turns green.
      if (state && state.converged && !wasConverged) {
        wasConverged = true;
        loopApi.pause();
        setPlayLabel();
        renderReadouts();
      }
      draw();
    }, { autoplay: true });

    function setPlayLabel() {
      btns.fvButtons["Play"].textContent = loopApi.playing ? "Pause" : "Play";
    }
    function togglePlay() {
      if (loopApi.playing) loopApi.pause();
      else loopApi.play();
      setPlayLabel();
    }

    if (loopApi.reduced) {
      btns.fvButtons["Play"].disabled = true;
      btns.fvButtons["Play"].title = "Reduced motion is on — use Step";
      btns.fvButtons["Step"].classList.add("fv-primary");
    }

    FV.onThemeChange(function () { heatDirty = true; draw(); });

    // first paint: build the engine (reduced motion pre-runs ~25 generations
    // inside rebuild(), so its single static frame is already meaningful).
    rebuild();
    draw();
    setPlayLabel();
  }
})();
