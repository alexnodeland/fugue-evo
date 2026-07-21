// docs/viz/landscape.js — populations on real benchmark landscapes.
//
// Two widgets over the REAL fugue-evo crate compiled to WebAssembly
// (crates/fugue-evo-wasm/src/explore.rs). No algorithm math lives here:
// wasm computes, this file only draws.
//
//   "ga-anatomy"    — the anatomy of one GA generation on a 2-D benchmark,
//                     animated phase by phase: selection (green parent rings,
//                     pair links) -> crossover (blue offspring, connector
//                     lines from the two parents) -> mutation (dots lerp to
//                     their final position with a violet displacement streak),
//                     then the offspring settle as the new population.
//                     Engine: ExploreGa (TournamentSelection + SbxCrossover +
//                     PolynomialMutation, elitism 1).
//   "umda-contract" — continuous UMDA (an EDA): sample, select the top
//                     fraction, refit a univariate-Gaussian product; the
//                     model is drawn as blue axis-aligned 1-sigma/2-sigma
//                     ellipses visibly sliding toward and contracting onto
//                     the basin. Engine: ExploreUmda.
//
// Fitness values streamed from wasm are RAW benchmark values (lower is
// better). Seeds are u64 and cross the boundary as BigInt. Self-contained
// IIFE; assumes fugue-viz.js and fugue-evo-wasm-loader.js loaded first.
(function () {
  "use strict";
  if (typeof window === "undefined" || !window.FugueViz) return;
  var FV = window.FugueViz;

  var NOTICE =
    "This figure runs the real fugue-evo crate compiled to WebAssembly — " +
    "the wasm package isn't available in this build.";

  var HEAT_N = 120; // landscape grid resolution (cells per axis), fixed

  // ---------------------------------------------------------------- helpers

  function toRgb(col) {
    col = (col || "").trim();
    var m = /^#?([0-9a-f]{6})$/i.exec(col);
    if (m) {
      var n = parseInt(m[1], 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    var r = /rgba?\(([^)]+)\)/.exec(col);
    if (r) {
      var p = r[1].split(",");
      return [parseInt(p[0], 10), parseInt(p[1], 10), parseInt(p[2], 10)];
    }
    return [128, 128, 128];
  }

  function showNotice(root, msg) {
    var d = document.createElement("div");
    d.className = "fv-pg-notice";
    d.textContent = msg;
    root.appendChild(d);
  }

  function el(tag, cls, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  }

  // A seed scrub inside the controls row: "SEED <draggable number>".
  function seedControl(controls, value, onInput) {
    var wrap = el("div", "fv-control", controls);
    var lab = el("span", "fv-control-label", wrap);
    lab.textContent = "SEED";
    var span = el("span", null, wrap);
    FV.scrub(span, {
      min: 1,
      max: 999,
      step: 1,
      value: value,
      fmt: function (v) {
        return String(v | 0);
      },
      onInput: onInput
    });
    return span;
  }

  // Recolor the cached landscape grid into an offscreen canvas. The grid is
  // row-major out[j*nx+i] with j indexing y from ymin up; canvas rows grow
  // downward, so rows are flipped here. Nonlinear ramp: alpha follows
  // 1 - sqrt(normalized) so the (lower-is-better) basins glow visibly.
  function buildHeat(off, grid, nx, ny, colHex) {
    off.width = nx;
    off.height = ny;
    var hctx = off.getContext("2d");
    var img = hctx.createImageData(nx, ny);
    var lo = Infinity,
      hi = -Infinity,
      k,
      v;
    for (k = 0; k < grid.length; k++) {
      v = grid[k];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    var span = hi - lo || 1;
    var rgb = toRgb(colHex);
    for (var j = 0; j < ny; j++) {
      for (var i = 0; i < nx; i++) {
        var t = (grid[j * nx + i] - lo) / span;
        var a = 0.6 * (1 - Math.sqrt(t));
        if (a < 0) a = 0;
        var p = ((ny - 1 - j) * nx + i) * 4; // y-flip
        img.data[p] = rgb[0];
        img.data[p + 1] = rgb[1];
        img.data[p + 2] = rgb[2];
        img.data[p + 3] = Math.round(255 * a);
      }
    }
    hctx.putImageData(img, 0, 0);
  }

  // Square data domain [lo,hi]^2 mapped into the canvas with axis insets.
  function mkPlot(w, h, dom) {
    var pad = { l: 40, r: 12, t: 10, b: 26 };
    var iw = Math.max(10, w - pad.l - pad.r);
    var ih = Math.max(10, h - pad.t - pad.b);
    return {
      ix: pad.l,
      iy: pad.t,
      iw: iw,
      ih: ih,
      sx: FV.scale(dom, [pad.l, pad.l + iw]),
      sy: FV.scale(dom, [pad.t + ih, pad.t])
    };
  }

  function dot(ctx, x, y, r, color, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.restore();
  }

  function ring(ctx, x, y, r, color, alpha, width) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 1.5;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
  }

  function seg(ctx, x0, y0, x1, y1, color, alpha, width, dash) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 1;
    ctx.lineCap = "round";
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.restore();
  }

  // Persistent best-so-far marker: soft coral glow + solid diamond.
  function bestMark(ctx, x, y, color) {
    ctx.save();
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 9, 0, 2 * Math.PI);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x + 5, y);
    ctx.lineTo(x, y + 5);
    ctx.lineTo(x - 5, y);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function fmtF(v) {
    if (!isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(2);
    return v.toFixed(3);
  }

  function smooth(p) {
    return p * p * (3 - 2 * p);
  }

  // ==========================================================================
  // Widget 1 — "ga-anatomy": one generation, phase by phase
  // ==========================================================================

  FV.register("ga-anatomy", function (root) {
    var p = FV.wasmReady || Promise.resolve(null);
    p.then(function (W) {
      if (!W) {
        root.setAttribute("data-fugue-backend", "none");
        showNotice(root, NOTICE);
        return;
      }
      root.setAttribute("data-fugue-backend", "wasm");
      try {
        gaInit(root, W);
      } catch (e) {
        try {
          console.error("[fugue-viz] ga-anatomy init failed", e);
        } catch (e2) { /* readout only */ }
        showNotice(root, "ga-anatomy failed to initialize: " + (e && e.message ? e.message : e));
      }
    });
  });

  function gaInit(root, W) {
    // -------------------------------------------------- per-instance options
    var seed0 = parseInt(root.getAttribute("data-seed") || "11", 10) || 11;
    var landscape = root.getAttribute("data-landscape") || "rastrigin";
    var popSize = parseInt(root.getAttribute("data-pop") || "40", 10) || 40;

    // Landscape metadata + precomputed heatmap grid (throws on a bad name).
    var info = JSON.parse(W.explore_landscape_info(landscape));
    var DOM = [info.lo, info.hi];
    var grid = W.explore_landscape_grid(
      landscape, info.lo, info.hi, info.lo, info.hi, HEAT_N, HEAT_N);
    var heatCv = document.createElement("canvas");
    var heatDirty = true;

    // ------------------------------------------------------------------ state
    var params = { seed: seed0, k: 3, pc: 0.9, pm: 0.6 };
    var engine = null;
    var cur = null; // parsed anatomy of the generation being animated
    var phaseT = 0; // seconds into the current generation's animation
    var SEL = 0.35,
      CROSS = 0.35,
      MUT = 0.5,
      TOTAL = SEL + CROSS + MUT;

    function advance() {
      cur = JSON.parse(engine.step());
    }

    // Rebuild the engine deterministically from the current seed + knobs.
    function rebuild() {
      engine = new W.ExploreGa(landscape, popSize, BigInt(params.seed));
      engine.setTournamentSize(params.k);
      engine.setRates(params.pc, params.pm);
      cur = null;
      phaseT = 0;
      advance();
      // Reduced motion: no phase animation — settle ~20 generations and show
      // the finished population as a single static frame.
      if (loopApi && loopApi.reduced) {
        for (var i = 0; i < 19; i++) advance();
        phaseT = TOTAL;
      }
    }

    // --------------------------------------------------------------- controls
    var controls = el("div", "fv-controls", root);

    FV.slider(controls, {
      label: "TOURNAMENT k",
      min: 1,
      max: 8,
      step: 1,
      value: params.k,
      fmt: function (v) {
        return String(v | 0);
      },
      onInput: function (v) {
        params.k = v | 0;
        engine.setTournamentSize(params.k);
      }
    });

    FV.slider(controls, {
      label: "CROSSOVER P",
      min: 0,
      max: 1,
      step: 0.01,
      value: params.pc,
      fmt: function (v) {
        return v.toFixed(2);
      },
      onInput: function (v) {
        params.pc = v;
        engine.setRates(params.pc, params.pm);
      }
    });

    FV.slider(controls, {
      label: "MUTATION P",
      min: 0,
      max: 1,
      step: 0.01,
      value: params.pm,
      fmt: function (v) {
        return v.toFixed(2);
      },
      onInput: function (v) {
        params.pm = v;
        engine.setRates(params.pc, params.pm);
      }
    });

    seedControl(controls, params.seed, function (v) {
      params.seed = v | 0;
      rebuild();
      renderReadouts();
      requestDraw();
    });

    var btns = FV.buttons(controls, [
      { label: "Play", title: "Animate generations", primary: true, onClick: function () { togglePlay(); } },
      { label: "Step", title: "One full generation (settled)", onClick: function () { loopApi.step(); } },
      { label: "Reset", title: "Rebuild from the current seed", onClick: function () { rebuild(); renderReadouts(); requestDraw(); } }
    ]);

    var cv = FV.canvas(root, { height: 340, onResize: function () { draw(); } });
    var ctx = cv.ctx;

    var instr = el("div", "fv-instruction", root);
    instr.textContent =
      "one generation in three phases — selection: green rings = chosen parents · " +
      "crossover: blue = pre-mutation offspring · mutation: violet streak = the nudge · " +
      "yellow ring = elite · coral = best-so-far";

    var readouts = el("div", "fv-readouts", root);
    var rGen = FV.readout(readouts, { label: "GEN" });
    var rBest = FV.readout(readouts, { label: "BEST f" });
    var rDiv = FV.readout(readouts, { label: "DIVERSITY" });

    function renderReadouts() {
      if (!cur) return;
      rGen.set(String(cur.gen));
      rBest.set(fmtF(cur.best[2]), "hot");
      rDiv.set(fmtF(cur.diversity));
    }

    // ------------------------------------------------------------------- draw
    function draw() {
      if (!cv) return; // onResize can fire during canvas() setup
      cv.clear();
      var th = FV.theme(),
        C = th.colors;
      var plot = mkPlot(cv.w, cv.h, DOM);

      if (heatDirty) {
        buildHeat(heatCv, grid, HEAT_N, HEAT_N, C.data);
        heatDirty = false;
      }
      ctx.drawImage(heatCv, plot.ix, plot.iy, plot.iw, plot.ih);
      FV.axes(ctx, {
        x: plot.ix, y: plot.iy, w: plot.iw, h: plot.ih,
        xscale: plot.sx, yscale: plot.sy, xlabel: "x", ylabel: "y", theme: th
      });
      if (!cur) return;

      // Everything population-related clips to the plot frame.
      ctx.save();
      ctx.beginPath();
      ctx.rect(plot.ix, plot.iy, plot.iw, plot.ih);
      ctx.clip();

      var pop = cur.population,
        off = cur.offspring,
        pre = cur.offspring_pre,
        pairs = cur.pairs;
      var isParent = {};
      var i, k, x, y;
      for (i = 0; i < cur.parents.length; i++) isParent[cur.parents[i]] = true;

      function PX(pt) { return plot.sx(pt[0]); }
      function PY(pt) { return plot.sy(pt[1]); }

      var phase, prog;
      if (phaseT < SEL) {
        phase = 0;
        prog = phaseT / SEL;
      } else if (phaseT < SEL + CROSS) {
        phase = 1;
        prog = (phaseT - SEL) / CROSS;
      } else {
        phase = 2;
        prog = Math.min(1, (phaseT - SEL - CROSS) / MUT);
      }

      if (phase === 0) {
        // SELECTION — parents ringed, non-parents dimmed, pair links flash.
        var linkA = Math.sin(Math.PI * Math.min(1, prog)) * 0.3;
        for (k = 0; k < pairs.length; k++) {
          var a = pop[pairs[k][0]],
            b = pop[pairs[k][1]];
          seg(ctx, PX(a), PY(a), PX(b), PY(b), C.ink, linkA, 1);
        }
        for (i = 0; i < pop.length; i++) {
          dot(ctx, PX(pop[i]), PY(pop[i]), 3.5, C.post, isParent[i] ? 0.95 : 0.28);
        }
        var ringA = Math.min(1, prog * 3);
        for (i = 0; i < pop.length; i++) {
          if (isParent[i]) ring(ctx, PX(pop[i]), PY(pop[i]), 6.5, C.post, ringA, 1.5);
        }
      } else if (phase === 1) {
        // CROSSOVER — blue pre-mutation offspring, fading parent connectors.
        for (i = 0; i < pop.length; i++) {
          dot(ctx, PX(pop[i]), PY(pop[i]), 3.5, C.post, isParent[i] ? 0.4 : 0.18);
        }
        var connA = 0.45 * (1 - prog);
        var preA = Math.min(1, prog * 2.5) * 0.95;
        for (k = 0; k < pre.length; k++) {
          var pr = pairs[Math.floor(k / 2)];
          if (pr && connA > 0.02) {
            var pa = pop[pr[0]],
              pb = pop[pr[1]];
            seg(ctx, PX(pa), PY(pa), PX(pre[k]), PY(pre[k]), C.prior, connA, 1);
            seg(ctx, PX(pb), PY(pb), PX(pre[k]), PY(pre[k]), C.prior, connA, 1);
          }
          dot(ctx, PX(pre[k]), PY(pre[k]), 3.5, C.prior, preA);
        }
      } else {
        // MUTATION — lerp pre -> final with a violet displacement streak.
        var e = smooth(prog);
        for (i = 0; i < pop.length; i++) {
          dot(ctx, PX(pop[i]), PY(pop[i]), 3, C.post, 0.12);
        }
        // offspring[0] is the carried-over elite; pre[k] maps to off[k+1].
        for (k = 0; k + 1 < off.length && k < pre.length; k++) {
          var sx0 = PX(pre[k]),
            sy0 = PY(pre[k]);
          var ex = PX(off[k + 1]),
            ey = PY(off[k + 1]);
          x = sx0 + (ex - sx0) * e;
          y = sy0 + (ey - sy0) * e;
          var dxp = ex - sx0,
            dyp = ey - sy0;
          if (dxp * dxp + dyp * dyp > 4 && e < 1) {
            seg(ctx, sx0, sy0, x, y, C.flow, 0.55 * (1 - e), 1.5);
          }
          // crossfade blue (pre) -> green (settled offspring)
          dot(ctx, x, y, 3.5, C.prior, 0.9 * (1 - e));
          dot(ctx, x, y, 3.5, C.post, 0.9 * e);
        }
        dot(ctx, PX(off[0]), PY(off[0]), 3.5, C.post, 0.95);
      }

      // The elite keeps its yellow ring throughout: the old population's elite
      // during selection/crossover, the carried-over copy once settled.
      var elitePt = phase === 2 ? off[0] : pop[cur.elite];
      ring(ctx, PX(elitePt), PY(elitePt), 8, C.data, 0.95, 1.6);

      // Best-so-far — persistent hot marker.
      bestMark(ctx, plot.sx(cur.best[0]), plot.sy(cur.best[1]), C.hot);

      ctx.restore(); // end plot clip
    }

    // schedule a single draw when paused (during play the loop already draws)
    var drawQueued = false;
    function requestDraw() {
      if (loopApi.playing) return;
      if (drawQueued) return;
      drawQueued = true;
      window.requestAnimationFrame(function () {
        drawQueued = false;
        draw();
      });
    }

    // ------------------------------------------------------------------- loop
    var loopApi = FV.loop(root, function (dt) {
      if (!engine) return;
      if (dt === 0) {
        // Step button / reduced motion: one full generation, settled.
        advance();
        phaseT = TOTAL;
        renderReadouts();
        draw();
        return;
      }
      phaseT += dt;
      if (phaseT >= TOTAL) {
        advance();
        phaseT = 0;
        renderReadouts();
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
      btns.fvButtons["Play"].title = "Reduced motion is on — use Step";
      btns.fvButtons["Step"].classList.add("fv-primary");
    }

    FV.onThemeChange(function () {
      heatDirty = true;
      draw();
    });

    // first paint
    rebuild();
    renderReadouts();
    draw();
    if (loopApi.playing) btns.fvButtons["Play"].textContent = "Pause";
  }

  // ==========================================================================
  // Widget 2 — "umda-contract": the learned Gaussian contracting onto the basin
  // ==========================================================================

  FV.register("umda-contract", function (root) {
    var p = FV.wasmReady || Promise.resolve(null);
    p.then(function (W) {
      if (!W) {
        root.setAttribute("data-fugue-backend", "none");
        showNotice(root, NOTICE);
        return;
      }
      root.setAttribute("data-fugue-backend", "wasm");
      try {
        umdaInit(root, W);
      } catch (e) {
        try {
          console.error("[fugue-viz] umda-contract init failed", e);
        } catch (e2) { /* readout only */ }
        showNotice(root, "umda-contract failed to initialize: " + (e && e.message ? e.message : e));
      }
    });
  });

  function umdaInit(root, W) {
    // -------------------------------------------------- per-instance options
    var seed0 = parseInt(root.getAttribute("data-seed") || "11", 10) || 11;
    var landscape = root.getAttribute("data-landscape") || "sphere";
    var ratio0 = parseFloat(root.getAttribute("data-ratio") || "0.3") || 0.3;
    var POP = 80;

    var info = JSON.parse(W.explore_landscape_info(landscape));
    var DOM = [info.lo, info.hi];
    var grid = W.explore_landscape_grid(
      landscape, info.lo, info.hi, info.lo, info.hi, HEAT_N, HEAT_N);
    var heatCv = document.createElement("canvas");
    var heatDirty = true;

    // ------------------------------------------------------------------ state
    var params = { seed: seed0, ratio: ratio0 };
    var engine = null;
    var cur = null;
    var pacer = FV.pace(4); // ~4 generations per second

    function advance() {
      cur = JSON.parse(engine.step());
    }

    function rebuild() {
      engine = new W.ExploreUmda(landscape, POP, params.ratio, BigInt(params.seed));
      cur = null;
      advance();
      if (loopApi && loopApi.reduced) {
        for (var i = 0; i < 19; i++) advance();
      }
    }

    // --------------------------------------------------------------- controls
    var controls = el("div", "fv-controls", root);

    FV.slider(controls, {
      label: "SELECT TOP",
      min: 0.1,
      max: 0.6,
      step: 0.01,
      value: params.ratio,
      fmt: function (v) {
        return Math.round(v * 100) + "%";
      },
      onInput: function (v) {
        params.ratio = v;
        rebuild(); // ratio is a constructor arg: rebuild with current seed
        renderReadouts();
        requestDraw();
      }
    });

    seedControl(controls, params.seed, function (v) {
      params.seed = v | 0;
      rebuild();
      renderReadouts();
      requestDraw();
    });

    var btns = FV.buttons(controls, [
      { label: "Play", title: "Run generations", primary: true, onClick: function () { togglePlay(); } },
      { label: "Reset", title: "Rebuild from the current seed", onClick: function () { rebuild(); renderReadouts(); requestDraw(); } }
    ]);

    var cv = FV.canvas(root, { height: 280, onResize: function () { draw(); } });
    var ctx = cv.ctx;

    var instr = el("div", "fv-instruction", root);
    instr.textContent =
      "green = sampled population (bright = the selected top fraction) · " +
      "blue ellipses = the learned Gaussian model (1σ solid, 2σ dashed) · coral = best-so-far";

    var readouts = el("div", "fv-readouts", root);
    var rGen = FV.readout(readouts, { label: "GEN" });
    var rBest = FV.readout(readouts, { label: "BEST f" });

    function renderReadouts() {
      if (!cur) return;
      rGen.set(String(cur.gen));
      rBest.set(fmtF(cur.best[2]), "hot");
    }

    // ------------------------------------------------------------------- draw
    function drawModelEllipse(plot, mx, my, sdx, sdy, color, alpha, width, dash) {
      var cx = plot.sx(mx),
        cy = plot.sy(my);
      var rx = Math.abs(plot.sx(mx + sdx) - cx);
      var ry = Math.abs(plot.sy(my + sdy) - cy);
      if (!isFinite(rx) || !isFinite(ry)) return;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.ellipse(cx, cy, Math.max(rx, 0.5), Math.max(ry, 0.5), 0, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.restore();
    }

    function draw() {
      if (!cv) return;
      cv.clear();
      var th = FV.theme(),
        C = th.colors;
      var plot = mkPlot(cv.w, cv.h, DOM);

      if (heatDirty) {
        buildHeat(heatCv, grid, HEAT_N, HEAT_N, C.data);
        heatDirty = false;
      }
      ctx.drawImage(heatCv, plot.ix, plot.iy, plot.iw, plot.ih);
      FV.axes(ctx, {
        x: plot.ix, y: plot.iy, w: plot.iw, h: plot.ih,
        xscale: plot.sx, yscale: plot.sy, xlabel: "x", ylabel: "y", theme: th
      });
      if (!cur) return;

      ctx.save();
      ctx.beginPath();
      ctx.rect(plot.ix, plot.iy, plot.iw, plot.ih);
      ctx.clip();

      // Sampled population, best-first: the selected top fraction bright, the
      // discarded rest faint.
      var pop = cur.population;
      for (var i = 0; i < pop.length; i++) {
        var sel = i < cur.selected;
        dot(ctx, plot.sx(pop[i][0]), plot.sy(pop[i][1]), sel ? 3.5 : 2.5, C.post, sel ? 0.95 : 0.2);
      }

      // The learned univariate-Gaussian product: axis-aligned 1σ / 2σ.
      var mx = cur.means[0],
        my = cur.means[1];
      var sdx = Math.sqrt(Math.max(0, cur.variances[0]));
      var sdy = Math.sqrt(Math.max(0, cur.variances[1]));
      drawModelEllipse(plot, mx, my, 2 * sdx, 2 * sdy, C.prior, 0.45, 1.2, [5, 4]);
      drawModelEllipse(plot, mx, my, sdx, sdy, C.prior, 0.95, 1.8, null);
      dot(ctx, plot.sx(mx), plot.sy(my), 3, C.prior, 0.95);

      // Best-so-far — persistent hot marker.
      bestMark(ctx, plot.sx(cur.best[0]), plot.sy(cur.best[1]), C.hot);

      ctx.restore(); // end plot clip
    }

    var drawQueued = false;
    function requestDraw() {
      if (loopApi.playing) return;
      if (drawQueued) return;
      drawQueued = true;
      window.requestAnimationFrame(function () {
        drawQueued = false;
        draw();
      });
    }

    // ------------------------------------------------------------------- loop
    var loopApi = FV.loop(root, function (dt) {
      if (!engine) return;
      if (dt === 0) {
        advance();
        renderReadouts();
        draw();
        return;
      }
      var n = pacer(dt);
      if (n > 0) {
        while (n-- > 0) advance();
        renderReadouts();
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
      btns.fvButtons["Play"].title = "Reduced motion is on — the settled model is shown";
    }

    FV.onThemeChange(function () {
      heatDirty = true;
      draw();
    });

    // first paint
    rebuild();
    renderReadouts();
    draw();
    if (loopApi.playing) btns.fvButtons["Play"].textContent = "Pause";
  }
})();
