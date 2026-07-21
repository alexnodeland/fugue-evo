// docs/viz/inline.js — the fugue-evo micro-widget family (ambient, inline).
//
// Five small figures embeddable in tutorial prose via
//   <div class="fugue-explorable fv-inline" data-viz="NAME" data-...></div>
//
//   mini-convergence  best-so-far raw fitness vs generation   (ExploreGa)
//   mini-diversity    diversity collapse, scrubbable k        (ExploreGa)
//   mini-sigma        CMA-ES step size σ vs generation        (ExploreCma)
//   mini-front        a Pareto front forming, objective space (ExploreNsga2)
//   mini-migration    island plateaus + migration drops       (ExploreIsland)
//
// WASM-ONLY compute: every number drawn here comes from the real fugue-evo
// crate (crates/fugue-evo-wasm/src/explore.rs) via FV.wasmReady. At init each
// widget records a bounded generation window from its seed — the engines are
// deterministic, so the recording IS the run — then loops a cursor over that
// recording forever (seamless replay from the same seed, zero per-frame wasm).
// Without the wasm pkg the widget renders a short notice instead of animating
// (data-fugue-backend="none"); with it, data-fugue-backend="wasm".
//
// Fitness values are RAW benchmark values, lower is better. Seeds are u64
// (BigInt at the boundary), default 11 via data-seed. Under reduced motion the
// full window is already computed, so the finished final frame is drawn once.
//
// Self-contained ES5 IIFE; assumes fugue-viz.js + fugue-evo-wasm-loader.js
// loaded first (book.toml order). Ambient chrome only: canvas ≤ 140px, one
// pause glyph, at most one scrub, in-canvas readouts.
(function () {
  "use strict";
  if (typeof window === "undefined" || !window.FugueViz) return;
  var FV = window.FugueViz;

  var HZ = 7;          // ~6–8 generations per second (FV.pace)
  var HOLD_TICKS = 9;  // linger on the final frame before replaying

  // ==========================================================================
  // Small helpers
  // ==========================================================================

  function clampNum(v, a, b) { return v < a ? a : v > b ? b : v; }

  var _coarse = null;
  function coarsePointer() {
    if (_coarse === null) _coarse = !!(window.matchMedia && window.matchMedia("(pointer: coarse)").matches);
    return _coarse;
  }
  function glyphClear() { return coarsePointer() ? 34 : 24; }

  function fmtNum(v) {
    if (v == null || !isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }
  function fmtSig(v) {
    if (v == null || !isFinite(v)) return "—";
    if (v >= 10) return v.toFixed(1);
    if (v >= 0.1) return v.toFixed(2);
    if (v >= 0.001) return v.toFixed(4);
    return v.toExponential(1);
  }

  function label(g, txt, x, y, c, role) {
    g.save(); g.fillStyle = role ? c[role] : c.ink; g.globalAlpha = role ? 0.95 : 0.65;
    g.font = "11px var(--mono-font, monospace)"; g.textBaseline = "top"; g.textAlign = "left";
    g.fillText(txt, x, y); g.restore();
  }
  function labelRight(g, txt, x, y, c, role) {
    g.save(); g.fillStyle = role ? c[role] : c.ink; g.globalAlpha = role ? 0.95 : 0.65;
    g.font = "11px var(--mono-font, monospace)"; g.textBaseline = "top"; g.textAlign = "right";
    g.fillText(txt, x, y); g.restore();
  }
  function baseline(g, x0, x1, y, c) {
    g.save(); g.strokeStyle = c.ink; g.globalAlpha = 0.22; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke(); g.restore();
  }
  function hotDot(g, x, y, c) {
    g.save(); g.globalAlpha = 0.28; g.fillStyle = c.hot;
    g.beginPath(); g.arc(x, y, 7, 0, 6.2832); g.fill();
    g.globalAlpha = 1;
    g.beginPath(); g.arc(x, y, 3, 0, 6.2832); g.fill();
    g.restore();
  }

  // ---- attribute parsing -----------------------------------------------------

  var LANDSCAPES = { sphere: 1, rastrigin: 1, rosenbrock: 1, ackley: 1, styblinski: 1 };
  var PROBLEMS = { zdt1: 1, zdt2: 1, zdt3: 1, schaffer: 1 };
  // Fallback bounds (from src/fitness/benchmarks.rs) should landscape_info fail.
  var BOUNDS = {
    sphere: [-5.12, 5.12], rastrigin: [-5.12, 5.12], rosenbrock: [-5, 10],
    ackley: [-32.768, 32.768], styblinski: [-5, 5]
  };

  function seedAttr(root) {
    var v = parseInt(root.getAttribute("data-seed"), 10);
    return v >= 0 ? v : 11;
  }
  function intAttr(root, name, dflt, lo, hi) {
    var v = parseInt(root.getAttribute(name), 10);
    if (!isFinite(v)) return dflt;
    return Math.round(clampNum(v, lo, hi));
  }
  function landAttr(root, dflt) {
    var v = (root.getAttribute("data-landscape") || "").toLowerCase();
    return LANDSCAPES[v] ? v : dflt;
  }
  function problemAttr(root, dflt) {
    var v = (root.getAttribute("data-problem") || "").toLowerCase();
    return PROBLEMS[v] ? v : dflt;
  }

  // ==========================================================================
  // WASM gating — every widget defers on FV.wasmReady; no JS fallback math.
  // ==========================================================================

  function registerWasm(name, initFn) {
    FV.register(name, function (root) {
      var p = FV.wasmReady || Promise.resolve(null);
      p.then(function (mod) {
        if (!mod) {
          root.setAttribute("data-fugue-backend", "none");
          var d = document.createElement("div");
          d.className = "fv-pg-notice";
          d.textContent = "This figure runs the real fugue-evo crate compiled to WebAssembly — the wasm package isn't available in this build.";
          root.appendChild(d);
          return;
        }
        root.setAttribute("data-fugue-backend", "wasm");
        try {
          initFn(root, mod);
        } catch (e) {
          if (typeof console !== "undefined") console.error("[fugue-viz] " + name + " init failed", e);
        }
      });
    });
  }

  function freeEngine(eng) {
    try { if (eng && typeof eng.free === "function") eng.free(); } catch (e) { /* best-effort */ }
  }

  // ==========================================================================
  // Recorders — run the real engine for a bounded window, keep only what the
  // canvas needs. Deterministic: same seed, same recording, forever.
  // ==========================================================================

  function recordGa(mod, landscape, seed, gens, k) {
    var eng = new mod.ExploreGa(landscape, 40, BigInt(seed));
    if (k > 0) eng.setTournamentSize(k);
    var frames = [], i, s;
    for (i = 0; i < gens; i++) {
      s = JSON.parse(eng.step());
      frames.push({ gen: s.gen, best: s.best[2], div: s.diversity });
    }
    freeEngine(eng);
    return frames;
  }

  function recordCma(mod, landscape, seed, gens) {
    var lo, hi;
    try {
      var info = JSON.parse(mod.explore_landscape_info(landscape));
      lo = info.lo; hi = info.hi;
    } catch (e) {
      var b = BOUNDS[landscape] || BOUNDS.rosenbrock;
      lo = b[0]; hi = b[1];
    }
    // Start in the far corner of the box; σ0 at the standard 0.3·range.
    var eng = new mod.ExploreCma(landscape, 0.7 * hi, 0.7 * lo, 0.3 * (hi - lo), 0, BigInt(seed));
    var frames = [], i, s;
    for (i = 0; i < gens; i++) {
      try { s = JSON.parse(eng.step()); } catch (e2) { break; }
      if (!(s.sigma > 0)) break;
      frames.push({ gen: s.gen, sigma: s.sigma, best: s.best[2] });
      if (s.converged) break;
    }
    freeEngine(eng);
    return frames;
  }

  function recordNsga(mod, problem, seed, gens) {
    var eng = new mod.ExploreNsga2(problem, 40, BigInt(seed));
    var frames = [], i, s;
    s = JSON.parse(eng.snapshot());          // gen 0: the unranked initial cloud
    frames.push({ gen: s.gen, pts: s.points });
    for (i = 0; i < gens; i++) {
      s = JSON.parse(eng.step());
      frames.push({ gen: s.gen, pts: s.points });
    }
    freeEngine(eng);
    return frames;
  }

  function recordIsland(mod, seed, interval, gens) {
    var eng = new mod.ExploreIsland("ackley", 4, 16, interval, "ring", BigInt(seed));
    var frames = [], i, j, s;
    for (i = 0; i < gens; i++) {
      s = JSON.parse(eng.step());
      var bests = [];
      for (j = 0; j < s.islands.length; j++) {
        var b = s.islands[j].best;
        bests.push(b ? b[2] : NaN);
      }
      frames.push({
        gen: s.gen,
        migrated: !!s.migrated,
        bests: bests,
        global: s.global_best ? s.global_best[2] : NaN
      });
    }
    freeEngine(eng);
    return frames;
  }

  // ==========================================================================
  // Mount scaffold — a cursor over a recording: ambient loop, pause glyph,
  // optional caption, reduced-motion static final frame, theme/resize repaint.
  // ==========================================================================

  function mount(root, spec) {
    var S = { frames: spec.frames || [], cursor: 0, hold: 0 };
    var ready = false;
    var cv = FV.canvas(root, {
      height: spec.height || 120,
      onResize: function () { if (ready) render(); }
    });
    var g = cv.ctx;

    var capText = root.getAttribute("data-caption");
    if (capText) {
      var cap = document.createElement("div");
      cap.className = "fv-caption";
      cap.textContent = capText;
      root.appendChild(cap);
    }

    var pacer = FV.pace(spec.hz || HZ);
    function render() {
      cv.clear();
      try { spec.draw(g, S, cv.w, cv.h, FV.theme().colors); } catch (e) { /* keep the page quiet */ }
    }
    function advance() {
      if (S.frames.length < 2) return;
      if (S.cursor >= S.frames.length - 1) {
        S.hold++;
        if (S.hold > HOLD_TICKS) { S.hold = 0; S.cursor = 0; } // replay: same seed, same recording
      } else {
        S.cursor++;
      }
    }
    var loopApi = FV.loop(root, function (dt) {
      for (var n = pacer(dt); n-- > 0;) advance();
      render();
    });

    var glyph = document.createElement("button");
    glyph.type = "button";
    glyph.className = "fv-glyph";
    glyph.setAttribute("aria-label", "Pause or play this animation");
    function glyphUpdate() {
      glyph.textContent = loopApi.playing ? "‖" : "▶";
      glyph.title = loopApi.playing ? "Pause" : "Play";
    }
    glyph.addEventListener("click", function () {
      if (loopApi.playing) loopApi.pause(); else loopApi.play();
      glyphUpdate(); render();
    });
    root.appendChild(glyph);

    FV.onThemeChange(function () { render(); });

    ready = true;
    if (loopApi.reduced) {
      // Full window already computed — show the finished picture, no motion.
      S.cursor = Math.max(0, S.frames.length - 1);
      render();
      glyph.style.display = "none";
    } else {
      render();
      loopApi.play();
      glyphUpdate();
    }

    return {
      root: root,
      setFrames: function (frames) {
        S.frames = frames || [];
        S.hold = 0;
        S.cursor = loopApi.reduced ? Math.max(0, S.frames.length - 1) : 0;
        render();
      }
    };
  }

  // Shared plot inset for the series widgets.
  function inset(w, h) { return { x0: 12, x1: w - 10, y0: 16, y1: h - 10 }; }

  // ==========================================================================
  // 1. mini-convergence — best-so-far raw fitness falling, sqrt-scaled.
  //    data-landscape (sphere|rastrigin|rosenbrock|ackley|styblinski, default
  //    rastrigin), data-seed (default 11).
  // ==========================================================================

  registerWasm("mini-convergence", function (root, mod) {
    var landscape = landAttr(root, "rastrigin");
    var seed = seedAttr(root);
    var frames = recordGa(mod, landscape, seed, 80, 0);

    mount(root, {
      height: 120,
      frames: frames,
      draw: function (g, S, w, h, c) {
        var F = S.frames, N = F.length;
        if (!N) return;
        var r = inset(w, h);
        // sqrt-scaled y over values ≥ 0 (best-so-far is monotone: max at F[0];
        // shift only if a landscape's raw optimum dips below zero).
        var base = Math.min(0, F[N - 1].best);
        function tf(v) { return Math.sqrt(Math.max(v - base, 0)); }
        var ymax = tf(F[0].best) || 1;
        var xs = FV.scale([0, N - 1], [r.x0, r.x1]);
        var ys = FV.scale([0, ymax * 1.08], [r.y1, r.y0]);
        baseline(g, r.x0, r.x1, r.y1, c);
        g.save();
        g.beginPath(); g.rect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0); g.clip();
        var pts = [], i;
        for (i = 0; i <= S.cursor && i < N; i++) pts.push([xs(i), ys(tf(F[i].best))]);
        FV.curve(g, pts, { color: c.post, width: 1.6 });
        var cur = F[Math.min(S.cursor, N - 1)];
        hotDot(g, xs(Math.min(S.cursor, N - 1)), ys(tf(cur.best)), c);
        g.restore();
        label(g, landscape + " · gen " + cur.gen, r.x0, 2, c);
        labelRight(g, "best " + fmtNum(cur.best), w - 10 - glyphClear(), 2, c, "post");
      }
    });
  });

  // ==========================================================================
  // 2. mini-diversity — population diversity collapsing under selection
  //    pressure; tournament size k scrubbable in the caption row.
  //    data-seed (default 11), data-k (1..8, default 3).
  // ==========================================================================

  registerWasm("mini-diversity", function (root, mod) {
    var seed = seedAttr(root);
    var k = intAttr(root, "data-k", 3, 1, 8);

    var api = mount(root, {
      height: 120,
      frames: recordGa(mod, "rastrigin", seed, 80, k),
      draw: function (g, S, w, h, c) {
        var F = S.frames, N = F.length;
        if (!N) return;
        var r = inset(w, h);
        var ymax = 0, i;
        for (i = 0; i < N; i++) if (F[i].div > ymax) ymax = F[i].div;
        if (!(ymax > 0)) ymax = 1;
        var xs = FV.scale([0, N - 1], [r.x0, r.x1]);
        var ys = FV.scale([0, ymax * 1.1], [r.y1, r.y0]);
        baseline(g, r.x0, r.x1, r.y1, c);
        g.save();
        g.beginPath(); g.rect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0); g.clip();
        var pts = [];
        for (i = 0; i <= S.cursor && i < N; i++) pts.push([xs(i), ys(F[i].div)]);
        FV.curve(g, pts, { color: c.flow, width: 1.6 });
        var cur = F[Math.min(S.cursor, N - 1)];
        hotDot(g, xs(Math.min(S.cursor, N - 1)), ys(cur.div), c);
        g.restore();
        label(g, "gen " + cur.gen, r.x0, 2, c);
        labelRight(g, "diversity " + fmtNum(cur.div), w - 10 - glyphClear(), 2, c, "flow");
      }
    });

    // The one control: a Victor scrub for tournament size k in the caption row.
    // Scrubbing rebuilds the whole recording from the same seed — bigger k,
    // faster collapse: selection pressure in one picture.
    var row = document.createElement("div");
    row.className = "fv-caption";
    row.appendChild(document.createTextNode("selection pressure: tournament k = "));
    var span = document.createElement("span");
    row.appendChild(span);
    row.appendChild(document.createTextNode(" — bigger k collapses diversity faster"));
    root.appendChild(row);
    FV.scrub(span, {
      min: 1, max: 8, step: 1, value: k,
      onInput: function (v) {
        api.setFrames(recordGa(mod, "rastrigin", seed, 80, v));
      }
    });
  });

  // ==========================================================================
  // 3. mini-sigma — CMA-ES global step size σ shrinking, log-scaled.
  //    data-landscape (default rosenbrock), data-seed (default 11).
  // ==========================================================================

  registerWasm("mini-sigma", function (root, mod) {
    var landscape = landAttr(root, "rosenbrock");
    var seed = seedAttr(root);
    var frames = recordCma(mod, landscape, seed, 100);

    mount(root, {
      height: 120,
      frames: frames,
      draw: function (g, S, w, h, c) {
        var F = S.frames, N = F.length;
        if (!N) return;
        var r = inset(w, h);
        var lo = Infinity, hi = -Infinity, i, s;
        for (i = 0; i < N; i++) {
          s = F[i].sigma;
          if (s < lo) lo = s;
          if (s > hi) hi = s;
        }
        if (!(lo > 0)) lo = 1e-12;
        if (!(hi > lo)) hi = lo * 10;
        var xs = FV.scale([0, N - 1], [r.x0, r.x1]);
        var ys = FV.scale([Math.log(lo) - 0.15, Math.log(hi) + 0.15], [r.y1, r.y0]);
        baseline(g, r.x0, r.x1, r.y1, c);
        g.save();
        g.beginPath(); g.rect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0); g.clip();
        var pts = [];
        for (i = 0; i <= S.cursor && i < N; i++) pts.push([xs(i), ys(Math.log(F[i].sigma))]);
        FV.curve(g, pts, { color: c.prior, width: 1.6 });
        var cur = F[Math.min(S.cursor, N - 1)];
        hotDot(g, xs(Math.min(S.cursor, N - 1)), ys(Math.log(cur.sigma)), c);
        g.restore();
        label(g, landscape + " · gen " + cur.gen + " · log σ", r.x0, 2, c);
        labelRight(g, "σ " + fmtSig(cur.sigma), w - 10 - glyphClear(), 2, c, "prior");
      }
    });
  });

  // ==========================================================================
  // 4. mini-front — NSGA-II objective-space scatter; the Pareto front forms
  //    against the analytic reference. data-problem (zdt1|zdt2|zdt3|schaffer,
  //    default zdt1), data-seed (default 11). Fixed axes f1∈[0,1.05], f2∈[0,4.5].
  // ==========================================================================

  function refFront(problem) {
    // Analytic reference fronts where a single smooth curve exists on these
    // axes. zdt1: f2 = 1 − sqrt(f1); zdt2: f2 = 1 − f1².
    if (problem !== "zdt1" && problem !== "zdt2") return null;
    var pts = [], i, t;
    for (i = 0; i <= 80; i++) {
      t = i / 80;
      pts.push([t, problem === "zdt1" ? 1 - Math.sqrt(t) : 1 - t * t]);
    }
    return pts;
  }

  registerWasm("mini-front", function (root, mod) {
    var problem = problemAttr(root, "zdt1");
    var seed = seedAttr(root);
    var frames = recordNsga(mod, problem, seed, 60);
    var ref = refFront(problem);

    mount(root, {
      height: 140,
      frames: frames,
      draw: function (g, S, w, h, c) {
        var F = S.frames, N = F.length;
        if (!N) return;
        var r = inset(w, h);
        var xs = FV.scale([0, 1.05], [r.x0, r.x1]);
        var ys = FV.scale([0, 4.5], [r.y1, r.y0]);
        // frame
        g.save(); g.strokeStyle = c.ink; g.globalAlpha = 0.22; g.lineWidth = 1;
        g.strokeRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0); g.restore();
        g.save();
        g.beginPath(); g.rect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0); g.clip();
        // faint analytic reference front (data role)
        if (ref) {
          var rp = [], i;
          for (i = 0; i < ref.length; i++) rp.push([xs(ref[i][0]), ys(ref[i][1])]);
          g.save(); g.globalAlpha = 0.45;
          FV.curve(g, rp, { color: c.data, width: 1.2, dash: [3, 3] });
          g.restore();
        }
        var cur = F[Math.min(S.cursor, N - 1)];
        var pts = cur.pts, j, p;
        // dominated / unranked points first — faint chrome
        g.save(); g.fillStyle = c.ink; g.globalAlpha = 0.25;
        for (j = 0; j < pts.length; j++) {
          p = pts[j];
          if (p[2] === 0) continue;
          g.beginPath(); g.arc(xs(p[0]), ys(p[1]), 1.8, 0, 6.2832); g.fill();
        }
        g.restore();
        // rank-0 (the front) — post
        g.save(); g.fillStyle = c.post; g.globalAlpha = 0.9;
        for (j = 0; j < pts.length; j++) {
          p = pts[j];
          if (p[2] !== 0) continue;
          g.beginPath(); g.arc(xs(p[0]), ys(p[1]), 2.4, 0, 6.2832); g.fill();
        }
        g.restore();
        g.restore(); // end clip
        label(g, problem + " · f1 × f2", r.x0, 2, c);
        labelRight(g, "gen " + cur.gen, w - 10 - glyphClear(), 2, c, "post");
      }
    });
  });

  // ==========================================================================
  // 5. mini-migration — 4 ring islands on ackley: separated plateaus, violet
  //    migration ticks, drops after each tick. data-seed (default 11),
  //    data-interval (1..40, default 8).
  // ==========================================================================

  registerWasm("mini-migration", function (root, mod) {
    var seed = seedAttr(root);
    var interval = intAttr(root, "data-interval", 8, 1, 40);
    var frames = recordIsland(mod, seed, interval, 80);
    var ALPHAS = [0.35, 0.5, 0.7, 0.9];

    mount(root, {
      height: 130,
      frames: frames,
      draw: function (g, S, w, h, c) {
        var F = S.frames, N = F.length;
        if (!N) return;
        var r = inset(w, h);
        // sqrt-scaled best fitness (ackley raw ≥ 0) so late drops stay visible.
        function tf(v) { return Math.sqrt(Math.max(v, 0)); }
        var ymax = 0, i, j, v;
        for (i = 0; i < N; i++) {
          for (j = 0; j < F[i].bests.length; j++) {
            v = F[i].bests[j];
            if (isFinite(v) && tf(v) > ymax) ymax = tf(v);
          }
        }
        if (!(ymax > 0)) ymax = 1;
        var xs = FV.scale([0, N - 1], [r.x0, r.x1]);
        var ys = FV.scale([0, ymax * 1.08], [r.y1, r.y0]);
        baseline(g, r.x0, r.x1, r.y1, c);
        g.save();
        g.beginPath(); g.rect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0); g.clip();
        // migration ticks (flow), revealed with the timeline
        g.save(); g.strokeStyle = c.flow; g.globalAlpha = 0.4; g.lineWidth = 1;
        for (i = 0; i <= S.cursor && i < N; i++) {
          if (!F[i].migrated) continue;
          g.beginPath(); g.moveTo(xs(i), r.y0); g.lineTo(xs(i), r.y1); g.stroke();
        }
        g.restore();
        // 4 per-island best traces (post, layered alphas)
        for (j = 0; j < 4; j++) {
          var pts = [];
          for (i = 0; i <= S.cursor && i < N; i++) {
            v = F[i].bests[j];
            pts.push([xs(i), isFinite(v) ? ys(tf(v)) : NaN]);
          }
          g.save(); g.globalAlpha = ALPHAS[j];
          FV.curve(g, pts, { color: c.post, width: 1.2 });
          g.restore();
        }
        var cur = F[Math.min(S.cursor, N - 1)];
        if (isFinite(cur.global)) {
          var gi = Math.min(S.cursor, N - 1);
          hotDot(g, xs(gi), ys(tf(cur.global)), c);
        }
        g.restore(); // end clip
        label(g, "4 islands · ring · gen " + cur.gen, r.x0, 2, c);
        labelRight(g, "best " + fmtNum(cur.global), w - 10 - glyphClear(), 2, c, "post");
      }
    });
  });
})();
