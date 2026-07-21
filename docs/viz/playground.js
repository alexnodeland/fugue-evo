/* Playground — pick an algorithm, pick a problem, and watch the REAL
 * fugue-evo crate run it in the browser (WebAssembly build of the actual
 * crate; see fugue-evo-wasm-loader.js).
 *
 * Every generation on this page is computed by the crate's own engines
 * (crates/fugue-evo-wasm/src/explore.rs): ExploreGa, ExploreCma,
 * ExploreNsga2, ExploreIsland, ExploreUmda. JavaScript only draws. Without
 * the wasm package the widget explains itself instead of pretending.
 *
 * Conventions (mirrors the engine side):
 *  - step()/snapshot() return JSON strings — parsed here;
 *  - seeds are u64 and cross the boundary as BigInt;
 *  - fitness values are RAW benchmark values, lower is better (NSGA-II:
 *    both objectives minimized).
 */
(function () {
  "use strict";
  if (typeof window === "undefined" || !window.FugueViz) return;
  var FV = window.FugueViz;

  var LANDSCAPES = ["sphere", "rastrigin", "rosenbrock", "ackley", "styblinski"];
  var MO_PROBLEMS = ["zdt1", "zdt2", "zdt3", "schaffer"];
  var ALGOS = [
    { label: "SimpleGA", key: "ga" },
    { label: "CMA-ES", key: "cma" },
    { label: "NSGA-II", key: "nsga" },
    { label: "Island model", key: "island" },
    { label: "UMDA", key: "umda" }
  ];
  var TOPOLOGIES = ["ring", "full", "random", "star"];

  var N_ISLANDS = 4;    // fixed: the canvas shows four quadrants
  var HWIN = 200;       // rolling convergence window (generations)
  var TRAILCAP = 400;   // CMA-ES mean-trail length
  var GRID_N = 96;      // landscape heat grid resolution (per axis)
  var MIG_DUR = 0.7;    // migration flash lifetime (seconds, wall clock)
  var STEPCAP = 10;     // max generations advanced per animation frame
  var BATCH = 40;       // reduced-motion: one synchronous batch of this size
  var PAD = { l: 44, r: 14, t: 10, b: 26 };

  // Fixed objective-space windows so the axes never jump while a front forms.
  // Early random points can fall outside; everything dynamic is clipped.
  var MO_DOMAIN = {
    zdt1: { x: [0, 1], y: [0, 7] },
    zdt2: { x: [0, 1], y: [0, 7] },
    zdt3: { x: [0, 1], y: [-1, 7] },
    schaffer: { x: [0, 4.5], y: [0, 4.5] }
  };

  FV.register("evo-playground", function (root) {
    (FV.wasmReady || Promise.resolve(null)).then(function (W) {
      init(root, W);
    });
  });

  function init(root, W) {
    root.setAttribute("data-fugue-backend", W ? "wasm" : "none");
    if (!W) {
      var note = mk("div", "fv-pg-notice", root);
      note.innerHTML =
        "The playground runs the <em>real</em> fugue-evo crate compiled to " +
        "WebAssembly, and that package isn’t available in this build. On the " +
        "deployed site it loads automatically; for a local build, run " +
        "<code>wasm-pack build crates/fugue-evo-wasm --target web --release</code> " +
        "and copy <code>pkg/</code> next to the book (see the Docs workflow).";
      return;
    }

    // ---- State -----------------------------------------------------------
    var seed0 = parseInt(root.getAttribute("data-seed") || "11", 10);
    if (!isFinite(seed0) || seed0 < 1) seed0 = 11;
    var S = {
      algo: "ga",
      landscape: "rastrigin", // problem for the landscape algorithms
      mo: "zdt1",             // problem for NSGA-II
      seed: seed0,
      speed: 8,               // generations per second while playing
      P: {
        gaPop: 60, gaTourK: 3, gaPm: 0.6, gaPc: 0.9,
        cmaSigma: 1.5, cmaLambda: 0,
        nsgaPop: 60,
        isPop: 24, isInterval: 8, isTopo: "ring",
        umdaPop: 80, umdaRatio: 0.3
      },
      eng: null,
      last: null,     // parsed JSON of the latest step()/snapshot()
      gens: 0,
      history: [],    // [gen, value] — best raw f, or rank-0 fraction (NSGA)
      trail: [],      // CMA-ES mean trail in data coords
      migLife: 0,     // island migration flash (1 -> 0, time-based)
      cmaStart: null, // {x, y} or null -> default (0.7·hi, 0.7·lo)
      err: ""
    };

    var infoCache = {}; // landscape name -> {name, lo, hi, optimum_fitness, optimum}
    var gridCache = {}; // landscape name -> {vals, min, max}

    // ---- Controls row 1 --------------------------------------------------
    var controls = mk("div", "fv-controls", root);
    var algoSel = selectControl(controls, "algorithm", ALGOS.map(function (a) { return a.label; }));
    var probSel = selectControl(controls, "problem", LANDSCAPES);
    scrubControl(controls, "seed", {
      min: 1, max: 9999, step: 1, value: S.seed,
      fmt: function (v) { return String(v | 0); },
      onInput: function (v) { S.seed = v | 0; rebuild(); }
    });
    var btns = FV.buttons(controls, [
      { label: "Run", primary: true, title: "Run generations continuously", onClick: onRun },
      { label: "Pause", onClick: function () { anim.pause(); } },
      { label: "Step", title: "Advance one generation", onClick: function () { anim.step(); } },
      { label: "Reset", title: "Rebuild from the current seed", onClick: function () { anim.pause(); rebuild(); } }
    ]);
    FV.slider(controls, {
      label: "speed", min: 1, max: 30, step: 1, value: S.speed,
      fmt: function (v) { return (v | 0) + " gen/s"; },
      onInput: function (v) { S.speed = v | 0; }
    });

    // ---- Controls row 2 (algorithm-specific, rebuilt on algo change) -----
    var row2 = mk("div", "fv-controls", root);
    var feedback = mk("div", "fv-pg-feedback", root);

    // ---- Canvases --------------------------------------------------------
    var mainCv = FV.canvas(root, { height: 380, onResize: draw });
    var instr = mk("div", "fv-instruction", root);
    var stripCv = FV.canvas(root, { height: 120, onResize: draw });

    // ---- Readouts + badge ------------------------------------------------
    var readoutWrap = mk("div", "fv-readouts", root);
    var RO = {};
    var badge = mk("div", "fv-hint fv-pg-badge", root);
    badge.textContent =
      "every generation on this page is computed by the actual fugue-evo " +
      "crate (fugue-evo-wasm " + safeVersion(W) + ") — same operators, same " +
      "seeds, in your browser";

    // ---- Landscape metadata / heatmap ------------------------------------
    function infoFor(name) {
      if (!infoCache[name]) {
        try {
          infoCache[name] = JSON.parse(W.explore_landscape_info(name));
        } catch (e) {
          infoCache[name] = { name: name, lo: -5, hi: 5, optimum: null };
        }
      }
      return infoCache[name];
    }

    function gridFor(name) {
      if (!gridCache[name]) {
        var info = infoFor(name);
        var vals = null, mn = Infinity, mx = -Infinity;
        try {
          vals = W.explore_landscape_grid(name, info.lo, info.hi, info.lo, info.hi, GRID_N, GRID_N);
          for (var i = 0; i < vals.length; i++) {
            var v = vals[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
          }
        } catch (e) {
          vals = null;
        }
        gridCache[name] = { vals: vals, min: mn, max: mx };
      }
      return gridCache[name];
    }

    // Offscreen heat image at grid resolution, scaled at draw time. Rebuilt
    // only when the landscape or theme changes — never per frame.
    var heatCv = document.createElement("canvas");
    var heatKey = "";
    function ensureHeat(name) {
      var th = FV.theme();
      var key = name + "|" + (th.dark ? "d" : "l");
      if (key === heatKey) return;
      var g = gridFor(name);
      if (!g.vals) return;
      heatKey = key;
      heatCv.width = GRID_N;
      heatCv.height = GRID_N;
      var hctx = heatCv.getContext("2d");
      var img = hctx.createImageData(GRID_N, GRID_N);
      var rgb = toRgb(th.colors.data);
      var span = g.max - g.min || 1;
      var lspan = Math.log(1 + span);
      for (var j = 0; j < GRID_N; j++) {      // grid j indexes y from ymin up
        var row = GRID_N - 1 - j;             // canvas y grows downward
        for (var i = 0; i < GRID_N; i++) {
          // Log ramp so deep, narrow basins still read (Rosenbrock's corner
          // values dwarf the valley on a linear ramp). Basins glow in the
          // data role — the same ramp the tutorial widgets use.
          var t = Math.log(1 + (g.vals[j * GRID_N + i] - g.min)) / lspan;
          var off = (row * GRID_N + i) * 4;
          img.data[off] = rgb[0];
          img.data[off + 1] = rgb[1];
          img.data[off + 2] = rgb[2];
          img.data[off + 3] = Math.round(255 * 0.55 * (1 - t));
        }
      }
      hctx.putImageData(img, 0, 0);
    }

    function drawHeatInto(ctx, name, x, y, w, h) {
      ensureHeat(name);
      if (heatCv.width > 0 && heatKey.indexOf(name + "|") === 0) {
        ctx.drawImage(heatCv, x, y, w, h);
      }
    }

    // ---- Engine lifecycle ------------------------------------------------
    function cmaStart() {
      if (S.cmaStart) return S.cmaStart;
      var info = infoFor(S.landscape);
      return { x: 0.7 * info.hi, y: 0.7 * info.lo };
    }

    function lamSnap(v) {
      v = Math.round(v);
      if (v <= 0) return 0;      // 0 = library default λ
      return v < 6 ? 6 : v;
    }

    function fail(e) {
      S.err = String(e && e.message ? e.message : e);
      feedback.textContent = "✗ " + S.err;
      feedback.className = "fv-pg-feedback fv-pg-err";
    }

    function clearErr() {
      S.err = "";
      feedback.textContent = "";
      feedback.className = "fv-pg-feedback";
    }

    // Rebuild the engine from the current seed. Same seed -> identical replay
    // (the engines never touch OS entropy).
    function rebuild() {
      S.eng = null;
      S.last = null;
      S.gens = 0;
      S.history = [];
      S.trail = [];
      S.migLife = 0;
      clearErr();
      var seed = BigInt(Math.round(S.seed));
      try {
        if (S.algo === "ga") {
          S.eng = new W.ExploreGa(S.landscape, S.P.gaPop, seed);
          S.eng.setTournamentSize(S.P.gaTourK);
          S.eng.setRates(S.P.gaPc, S.P.gaPm);
        } else if (S.algo === "cma") {
          var st = cmaStart();
          S.eng = new W.ExploreCma(S.landscape, st.x, st.y, S.P.cmaSigma, lamSnap(S.P.cmaLambda), seed);
        } else if (S.algo === "nsga") {
          S.eng = new W.ExploreNsga2(S.mo, S.P.nsgaPop, seed);
        } else if (S.algo === "island") {
          S.eng = new W.ExploreIsland(S.landscape, N_ISLANDS, S.P.isPop, S.P.isInterval, S.P.isTopo, seed);
        } else {
          S.eng = new W.ExploreUmda(S.landscape, S.P.umdaPop, S.P.umdaRatio, seed);
        }
      } catch (e) {
        fail(e);
        draw();
        return;
      }
      if (S.algo === "nsga") {
        // snapshot() paints the unranked initial population without stepping.
        S.last = JSON.parse(S.eng.snapshot());
        S.gens = S.last.gen;
        pushHistory();
      } else {
        stepOnce(); // first generation, so the first paint shows a population
      }
      draw();
    }

    function stepOnce() {
      if (!S.eng) return;
      var js;
      try {
        js = S.eng.step();
      } catch (e) {
        fail(e);
        anim.pause();
        return;
      }
      S.last = JSON.parse(js);
      S.gens = S.last.gen;
      if (S.algo === "cma") {
        S.trail.push([S.last.mean[0], S.last.mean[1]]);
        if (S.trail.length > TRAILCAP) S.trail.shift();
      }
      if (S.algo === "island" && S.last.migrated) S.migLife = 1;
      pushHistory();
    }

    function pushHistory() {
      var v, i;
      if (S.algo === "nsga") {
        var pts = S.last.points, n0 = 0;
        for (i = 0; i < pts.length; i++) if (pts[i][2] === 0) n0++;
        v = pts.length ? n0 / pts.length : 0;
      } else if (S.algo === "island") {
        v = S.last.global_best ? S.last.global_best[2] : NaN;
      } else {
        v = S.last.best[2];
      }
      S.history.push([S.gens, v]);
      if (S.history.length > HWIN) S.history.shift();
    }

    function onRun() {
      if (!S.eng) rebuild();
      if (!S.eng) return;
      if (anim.reduced) {
        // Reduced motion: one synchronous batch, one static frame.
        for (var i = 0; i < BATCH; i++) stepOnce();
        draw();
        return;
      }
      anim.play();
    }

    // ---- Animation loop --------------------------------------------------
    var accum = 0;
    var anim = FV.loop(root, function (dt) {
      if (dt === 0) { // Step button (or reduced-motion step)
        stepOnce();
        draw();
        return;
      }
      accum += dt * S.speed;
      var n = Math.floor(accum);
      if (n > 0) {
        accum -= n;
        if (n > STEPCAP) n = STEPCAP;
        for (var i = 0; i < n; i++) stepOnce();
      }
      if (S.migLife > 0) {
        S.migLife -= dt / MIG_DUR; // wall-clock decay, no frame-rate strobe
        if (S.migLife < 0) S.migLife = 0;
      }
      draw();
    });

    // ---- Algorithm-specific controls -------------------------------------
    function buildAlgoControls() {
      row2.innerHTML = "";
      if (S.algo === "ga") {
        FV.slider(row2, {
          label: "population", min: 20, max: 200, step: 5, value: S.P.gaPop,
          fmt: fmtInt,
          onInput: function (v) { S.P.gaPop = v | 0; rebuild(); }
        });
        FV.slider(row2, {
          label: "tournament k", min: 1, max: 8, step: 1, value: S.P.gaTourK,
          fmt: fmtInt,
          onInput: function (v) { // live: no rebuild, the engine takes it now
            S.P.gaTourK = v | 0;
            if (S.eng && S.algo === "ga") S.eng.setTournamentSize(v | 0);
          }
        });
        FV.slider(row2, {
          label: "mutation p", min: 0, max: 1, step: 0.01, value: S.P.gaPm,
          fmt: fmt2,
          onInput: function (v) { // live via setRates, crossover prob held at 0.9
            S.P.gaPm = v;
            if (S.eng && S.algo === "ga") S.eng.setRates(S.P.gaPc, v);
          }
        });
        instr.textContent = "green = population · yellow ring = elite · coral = best-so-far · lower is better";
      } else if (S.algo === "cma") {
        scrubControl(row2, "σ₀", {
          min: 0.1, max: 3, step: 0.05, value: S.P.cmaSigma,
          fmt: fmt2,
          onInput: function (v) { S.P.cmaSigma = v; rebuild(); }
        });
        FV.slider(row2, {
          label: "λ", min: 0, max: 40, step: 1, value: S.P.cmaLambda,
          fmt: function (v) { var s = lamSnap(v); return s === 0 ? "auto" : String(s); },
          onInput: function (v) { S.P.cmaLambda = v | 0; rebuild(); }
        });
        instr.textContent = "drag the ◯ start marker to restart the search there · blue = the adapting 1σ/2σ search ellipse · violet = mean trail";
      } else if (S.algo === "nsga") {
        FV.slider(row2, {
          label: "population", min: 20, max: 200, step: 5, value: S.P.nsgaPop,
          fmt: fmtInt,
          onInput: function (v) { S.P.nsgaPop = v | 0; rebuild(); }
        });
        instr.textContent = "objective space, both minimized · green = rank-0 front · yellow = analytic Pareto reference";
      } else if (S.algo === "island") {
        FV.slider(row2, {
          label: "island pop", min: 8, max: 64, step: 2, value: S.P.isPop,
          fmt: fmtInt,
          onInput: function (v) { S.P.isPop = v | 0; rebuild(); }
        });
        FV.slider(row2, {
          label: "interval", min: 2, max: 20, step: 1, value: S.P.isInterval,
          fmt: fmtInt,
          onInput: function (v) { S.P.isInterval = v | 0; rebuild(); }
        });
        var tb = FV.buttons(row2, TOPOLOGIES.map(function (t) {
          return {
            label: t,
            onClick: function () { S.P.isTopo = t; markTopo(tb); rebuild(); }
          };
        }));
        markTopo(tb);
        instr.textContent = "four islands evolve independently · violet flash = migration on the interval · yellow ring = island best";
      } else {
        FV.slider(row2, {
          label: "population", min: 20, max: 300, step: 5, value: S.P.umdaPop,
          fmt: fmtInt,
          onInput: function (v) { S.P.umdaPop = v | 0; rebuild(); }
        });
        FV.slider(row2, {
          label: "selection ratio", min: 0.1, max: 0.6, step: 0.01, value: S.P.umdaRatio,
          fmt: fmt2,
          onInput: function (v) { S.P.umdaRatio = v; rebuild(); }
        });
        instr.textContent = "blue = the learned univariate Gaussian model (1σ/2σ) · yellow = selected refit set · green = sampled population";
      }
    }

    function markTopo(tb) {
      for (var i = 0; i < TOPOLOGIES.length; i++) {
        var b = tb.fvButtons[TOPOLOGIES[i]];
        if (!b) continue;
        if (TOPOLOGIES[i] === S.P.isTopo) b.classList.add("fv-primary");
        else b.classList.remove("fv-primary");
      }
    }

    // ---- Readouts --------------------------------------------------------
    function buildReadouts() {
      readoutWrap.innerHTML = "";
      RO = {};
      RO.gen = FV.readout(readoutWrap, { label: "gen" });
      RO.evals = FV.readout(readoutWrap, { label: "evals" });
      RO.best = FV.readout(readoutWrap, { label: S.algo === "nsga" ? "front" : "best f" });
      if (S.algo === "cma") RO.extra = FV.readout(readoutWrap, { label: "σ" });
      else if (S.algo === "ga" || S.algo === "island") RO.extra = FV.readout(readoutWrap, { label: "diversity" });
      else if (S.algo === "umda") RO.extra = FV.readout(readoutWrap, { label: "model σx,σy" });
      else RO.extra = null;
    }

    function evalsCount() {
      if (S.algo === "cma") return S.last.evaluations;
      if (S.algo === "ga") return S.gens * S.P.gaPop;
      if (S.algo === "nsga") return S.gens * S.P.nsgaPop;
      if (S.algo === "island") return S.gens * N_ISLANDS * S.P.isPop;
      return S.gens * S.P.umdaPop;
    }

    function updateReadouts() {
      if (!S.last) {
        RO.gen.set("—");
        RO.evals.set("—");
        RO.best.set("—");
        if (RO.extra) RO.extra.set("—");
        return;
      }
      var i;
      RO.gen.set(String(S.gens));
      RO.evals.set(String(evalsCount()));
      if (S.algo === "nsga") {
        var pts = S.last.points, n0 = 0;
        for (i = 0; i < pts.length; i++) if (pts[i][2] === 0) n0++;
        RO.best.set(n0 + "/" + pts.length, "post");
      } else if (S.algo === "island") {
        RO.best.set(S.last.global_best ? fmtF(S.last.global_best[2]) : "—", "hot");
        var d = 0;
        for (i = 0; i < S.last.islands.length; i++) d += S.last.islands[i].diversity;
        RO.extra.set(fmtF(d / S.last.islands.length));
      } else {
        RO.best.set(fmtF(S.last.best[2]), "hot");
      }
      if (S.algo === "ga") RO.extra.set(fmtF(S.last.diversity));
      else if (S.algo === "cma") RO.extra.set(fmtF(S.last.sigma));
      else if (S.algo === "umda") {
        RO.extra.set(fmtF(Math.sqrt(S.last.variances[0])) + ", " + fmtF(Math.sqrt(S.last.variances[1])));
      }
    }

    // ---- Rendering: main canvas ------------------------------------------
    function singleGeom() {
      if (S.algo === "nsga" || S.algo === "island") return null;
      var info = infoFor(S.landscape);
      var w = mainCv.w - PAD.l - PAD.r, h = mainCv.h - PAD.t - PAD.b;
      if (w < 20 || h < 20) return null;
      return {
        info: info, w: w, h: h,
        sx: FV.scale([info.lo, info.hi], [PAD.l, PAD.l + w]),
        sy: FV.scale([info.lo, info.hi], [PAD.t + h, PAD.t])
      };
    }

    function draw() {
      if (!mainCv || !stripCv) return; // onResize can fire during canvas()
      drawMain();
      drawStrip();
      updateReadouts();
    }

    function drawMain() {
      var th = FV.theme();
      mainCv.clear();
      if (S.algo === "nsga") drawMo(th);
      else if (S.algo === "island") drawIslands(th);
      else drawSingle(th);
    }

    function drawSingle(th) {
      var g = singleGeom();
      if (!g) return;
      var ctx = mainCv.ctx, C = th.colors;
      drawHeatInto(ctx, S.landscape, PAD.l, PAD.t, g.w, g.h);
      FV.axes(ctx, { x: PAD.l, y: PAD.t, w: g.w, h: g.h, xscale: g.sx, yscale: g.sy, xlabel: "x", ylabel: "y", theme: th });
      // Known optimum — yellow reference cross.
      if (g.info.optimum) cross(ctx, g.sx(g.info.optimum[0]), g.sy(g.info.optimum[1]), 5, C.data, 0.9);

      // Everything driven by dynamics clips to the frame.
      ctx.save();
      ctx.beginPath();
      ctx.rect(PAD.l, PAD.t, g.w, g.h);
      ctx.clip();
      if (S.algo === "ga") drawGa(ctx, g, C);
      else if (S.algo === "cma") drawCma(ctx, g, C);
      else drawUmda(ctx, g, C);
      ctx.restore();

      cvLabel(mainCv, th, algoLabel() + " on " + S.landscape);
    }

    function drawGa(ctx, g, C) {
      if (!S.last) return;
      var pop = S.last.offspring, i; // the new population; entry 0 = elite
      for (i = 0; i < pop.length; i++) {
        dot(ctx, g.sx(pop[i][0]), g.sy(pop[i][1]), 2.5, C.post, 0.8);
      }
      if (pop.length > 0) {
        ring(ctx, g.sx(pop[0][0]), g.sy(pop[0][1]), 6, C.data, 0.95, 1.6); // elite
      }
      bestMarker(ctx, g.sx(S.last.best[0]), g.sy(S.last.best[1]), C);
    }

    function drawCma(ctx, g, C) {
      var st = cmaStart();
      if (S.last) {
        // Mean trail — violet motion.
        var tp = [], i;
        for (i = 0; i < S.trail.length; i++) tp.push([g.sx(S.trail[i][0]), g.sy(S.trail[i][1])]);
        ctx.save();
        ctx.globalAlpha = 0.65;
        FV.curve(ctx, tp, { color: C.flow, width: 1.4 });
        ctx.restore();

        // Population.
        var pop = S.last.population;
        for (i = 0; i < pop.length; i++) {
          dot(ctx, g.sx(pop[i][0]), g.sy(pop[i][1]), 2.5, C.post, 0.8);
        }

        // Search-distribution ellipse. cov is the UNSCALED C; the sampling
        // covariance is σ²·C, so axis radius_i = σ·sqrt(eigenvalue_i) and the
        // orientation is 0.5·atan2(2c01, c00−c11) (major axis).
        var cv0 = S.last.cov, a = cv0[0], b = cv0[1], c = cv0[3];
        var mid = (a + c) / 2;
        var disc = Math.sqrt(((a - c) / 2) * ((a - c) / 2) + b * b);
        var emax = Math.max(mid + disc, 0), emin = Math.max(mid - disc, 0);
        var ang = 0.5 * Math.atan2(2 * b, a - c);
        var sg = S.last.sigma;
        var r1 = sg * Math.sqrt(emax), r2 = sg * Math.sqrt(emin);
        var mx = S.last.mean[0], my = S.last.mean[1];
        ctx.save();
        ctx.globalAlpha = 0.9;
        FV.curve(ctx, ellipsePts(g.sx, g.sy, mx, my, r1, r2, ang), { color: C.prior, width: 2 });
        ctx.globalAlpha = 0.5;
        FV.curve(ctx, ellipsePts(g.sx, g.sy, mx, my, 2 * r1, 2 * r2, ang), { color: C.prior, width: 1.2, dash: [4, 3] });
        ctx.restore();
        dot(ctx, g.sx(mx), g.sy(my), 3, C.prior, 1);

        bestMarker(ctx, g.sx(S.last.best[0]), g.sy(S.last.best[1]), C);
      }

      // Draggable start marker (◯ + cross), halo while grabbed.
      var px = g.sx(st.x), py = g.sy(st.y);
      if (dragHandle && dragHandle.grabbed) FV.halo(ctx, px, py, 14, C.hot, 0.4);
      ring(ctx, px, py, 6, C.ink, 0.85, 1.4);
      cross(ctx, px, py, 3, C.ink, 0.85);
    }

    function drawUmda(ctx, g, C) {
      if (!S.last) return;
      // Population is sorted ascending by f; the first `selected` are the
      // refit set (yellow), the rest the discarded samples (faint green).
      var pop = S.last.population, nSel = S.last.selected, i;
      for (i = nSel; i < pop.length; i++) {
        dot(ctx, g.sx(pop[i][0]), g.sy(pop[i][1]), 2, C.post, 0.35);
      }
      for (i = 0; i < nSel && i < pop.length; i++) {
        dot(ctx, g.sx(pop[i][0]), g.sy(pop[i][1]), 2.5, C.data, 0.9);
      }
      // The learned axis-aligned Gaussian model — 1σ solid, 2σ dashed.
      var mx = S.last.means[0], my = S.last.means[1];
      var rx = Math.sqrt(Math.max(S.last.variances[0], 0));
      var ry = Math.sqrt(Math.max(S.last.variances[1], 0));
      ctx.save();
      ctx.globalAlpha = 0.9;
      FV.curve(ctx, ellipsePts(g.sx, g.sy, mx, my, rx, ry, 0), { color: C.prior, width: 2 });
      ctx.globalAlpha = 0.5;
      FV.curve(ctx, ellipsePts(g.sx, g.sy, mx, my, 2 * rx, 2 * ry, 0), { color: C.prior, width: 1.2, dash: [4, 3] });
      ctx.restore();
      dot(ctx, g.sx(mx), g.sy(my), 3, C.prior, 1);
      bestMarker(ctx, g.sx(S.last.best[0]), g.sy(S.last.best[1]), C);
    }

    function drawIslands(th) {
      var ctx = mainCv.ctx, C = th.colors;
      var info = infoFor(S.landscape);
      var w = mainCv.w - PAD.l - PAD.r, h = mainCv.h - PAD.t - PAD.b;
      if (w < 20 || h < 20) return;
      var gap = 10;
      var qw = (w - gap) / 2, qh = (h - gap) / 2;
      var globalF = S.last && S.last.global_best ? S.last.global_best[2] : NaN;
      for (var k = 0; k < N_ISLANDS; k++) {
        var qx = PAD.l + (k % 2) * (qw + gap);
        var qy = PAD.t + Math.floor(k / 2) * (qh + gap);
        var sx = FV.scale([info.lo, info.hi], [qx, qx + qw]);
        var sy = FV.scale([info.lo, info.hi], [qy + qh, qy]);
        drawHeatInto(ctx, S.landscape, qx, qy, qw, qh);
        FV.axes(ctx, { x: qx, y: qy, w: qw, h: qh, theme: th }); // frame only
        if (info.optimum) cross(ctx, sx(info.optimum[0]), sy(info.optimum[1]), 4, C.data, 0.7);

        ctx.save();
        ctx.beginPath();
        ctx.rect(qx, qy, qw, qh);
        ctx.clip();
        if (S.last && S.last.islands[k]) {
          var isl = S.last.islands[k];
          for (var i = 0; i < isl.population.length; i++) {
            dot(ctx, sx(isl.population[i][0]), sy(isl.population[i][1]), 2.2, C.post, 0.8);
          }
          if (isl.best) {
            ring(ctx, sx(isl.best[0]), sy(isl.best[1]), 5, C.data, 0.95, 1.5);
            if (isl.best[2] === globalF) {
              bestMarker(ctx, sx(isl.best[0]), sy(isl.best[1]), C);
            }
          }
        }
        // island index, faint
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = C.ink;
        ctx.font = "10px var(--mono-font, monospace)";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText("island " + (k + 1), qx + 5, qy + 4);
        ctx.restore();

        // Migration flash — violet frame, decaying on the wall clock.
        if (S.migLife > 0) {
          ctx.save();
          ctx.globalAlpha = 0.9 * S.migLife;
          ctx.strokeStyle = C.flow;
          ctx.lineWidth = 2.5;
          ctx.strokeRect(qx, qy, qw, qh);
          ctx.restore();
        }
      }
      cvLabel(mainCv, th, "Island model (" + S.P.isTopo + ") on " + S.landscape);
    }

    function drawMo(th) {
      var ctx = mainCv.ctx, C = th.colors;
      var dom = MO_DOMAIN[S.mo];
      var w = mainCv.w - PAD.l - PAD.r, h = mainCv.h - PAD.t - PAD.b;
      if (w < 20 || h < 20) return;
      var sx = FV.scale([dom.x[0], dom.x[1]], [PAD.l, PAD.l + w]);
      var sy = FV.scale([dom.y[0], dom.y[1]], [PAD.t + h, PAD.t]);
      FV.axes(ctx, { x: PAD.l, y: PAD.t, w: w, h: h, xscale: sx, yscale: sy, xlabel: "f1", ylabel: "f2", theme: th });

      ctx.save();
      ctx.beginPath();
      ctx.rect(PAD.l, PAD.t, w, h);
      ctx.clip();

      // Analytic reference front (yellow). zdt3's full curve is drawn faint —
      // only its non-dominated pieces are the true front.
      var ref = refFrontPts(S.mo), rp = [], i;
      for (i = 0; i < ref.length; i++) rp.push([sx(ref[i][0]), sy(ref[i][1])]);
      ctx.save();
      if (S.mo === "zdt3") {
        ctx.globalAlpha = 0.35;
        FV.curve(ctx, rp, { color: C.data, width: 1.2 });
      } else {
        ctx.globalAlpha = 0.85;
        FV.curve(ctx, rp, { color: C.data, width: 1.5, dash: [5, 4] });
      }
      ctx.restore();

      if (S.last) {
        var pts = S.last.points;
        // Scatter colored by rank: rank 0 green, deeper ranks fade into ink,
        // unranked (-1, the initial population) faint ink.
        for (i = 0; i < pts.length; i++) {
          var r = pts[i][2];
          if (r === 0) continue; // rank 0 drawn last, on top
          var alpha = r < 0 ? 0.3 : Math.max(0.12, 0.5 - 0.08 * r);
          dot(ctx, sx(pts[i][0]), sy(pts[i][1]), 2, C.ink, alpha);
        }
        // Evolved rank-0 front: dots + polyline (broken across zdt3's gaps).
        var r0 = [];
        for (i = 0; i < pts.length; i++) if (pts[i][2] === 0) r0.push(pts[i]);
        r0.sort(function (p, q) { return p[0] - q[0]; });
        var gapT = 0.12 * (dom.x[1] - dom.x[0]);
        var line = [];
        for (i = 0; i < r0.length; i++) {
          if (i > 0 && r0[i][0] - r0[i - 1][0] > gapT) line.push([NaN, NaN]);
          line.push([sx(r0[i][0]), sy(r0[i][1])]);
        }
        ctx.save();
        ctx.globalAlpha = 0.75;
        FV.curve(ctx, line, { color: C.post, width: 1.8 });
        ctx.restore();
        for (i = 0; i < r0.length; i++) {
          dot(ctx, sx(r0[i][0]), sy(r0[i][1]), 2.6, C.post, 0.95);
        }
      }
      ctx.restore();

      cvLabel(mainCv, th, "NSGA-II on " + S.mo + " (minimize both)");
    }

    // ---- Rendering: convergence strip ------------------------------------
    function drawStrip() {
      var th = FV.theme(), C = th.colors;
      var ctx = stripCv.ctx;
      stripCv.clear();
      var pad = { l: 44, r: 14, t: 8, b: 18 };
      var w = stripCv.w - pad.l - pad.r, h = stripCv.h - pad.t - pad.b;
      if (w < 20 || h < 20) return;
      var n = S.history.length;
      var g0 = n ? S.history[0][0] : 0;
      var g1 = n ? S.history[n - 1][0] : 1;
      if (g1 <= g0) g1 = g0 + 1;
      var xs = FV.scale([g0, g1], [pad.l, pad.l + w]);
      var i, pts = [];

      if (S.algo === "nsga") {
        var ys = FV.scale([0, 1.05], [pad.t + h, pad.t]);
        FV.axes(ctx, { x: pad.l, y: pad.t, w: w, h: h, xscale: xs, yscale: ys, theme: th });
        ctx.save();
        ctx.beginPath();
        ctx.rect(pad.l, pad.t, w, h);
        ctx.clip();
        for (i = 0; i < n; i++) pts.push([xs(S.history[i][0]), ys(S.history[i][1])]);
        FV.curve(ctx, pts, { color: C.post, width: 1.6 });
        ctx.restore();
        cvLabel(stripCv, th, "rank-0 fraction vs generation");
      } else {
        // Best raw fitness on a sign-preserving sqrt scale (Styblinski–Tang's
        // optimum is negative), rolling ~200-generation window.
        var lo = Infinity, hi = -Infinity;
        for (i = 0; i < n; i++) {
          var t = sqrtT(S.history[i][1]);
          if (isFinite(t)) {
            if (t < lo) lo = t;
            if (t > hi) hi = t;
          }
        }
        if (!isFinite(lo)) { lo = 0; hi = 1; }
        if (hi - lo < 1e-9) hi = lo + 1;
        var padv = (hi - lo) * 0.12;
        var ys2 = FV.scale([lo - padv, hi + padv], [pad.t + h, pad.t]);
        FV.axes(ctx, { x: pad.l, y: pad.t, w: w, h: h, xscale: xs, ylabel: "best f (√)", theme: th });
        ctx.save();
        ctx.beginPath();
        ctx.rect(pad.l, pad.t, w, h);
        ctx.clip();
        for (i = 0; i < n; i++) pts.push([xs(S.history[i][0]), ys2(sqrtT(S.history[i][1]))]);
        FV.curve(ctx, pts, { color: C.hot, width: 1.6 });
        ctx.restore();
        cvLabel(stripCv, th, "best raw fitness vs generation (sqrt scale)");
      }
    }

    function sqrtT(v) {
      return (v < 0 ? -1 : 1) * Math.sqrt(Math.abs(v));
    }

    // ---- CMA-ES start-marker drag (fullCapture:false — a swipe elsewhere
    // on the canvas still scrolls the page; only a hit claims the gesture).
    var dragHandle = FV.drag(mainCv.el, {
      inflate: 12,
      fullCapture: false,
      hitTest: function (x, y, slop) {
        if (S.algo !== "cma") return null;
        var g = singleGeom();
        if (!g) return null;
        var st = cmaStart();
        var dx = g.sx(st.x) - x, dy = g.sy(st.y) - y;
        var r = Math.max(14, slop || 0);
        return dx * dx + dy * dy <= r * r ? "start" : null;
      },
      onDrag: function (t, x, y) {
        var g = singleGeom();
        if (!g) return;
        var nx = clampNum(g.sx.invert(x), g.info.lo, g.info.hi);
        var ny = clampNum(g.sy.invert(y), g.info.lo, g.info.hi);
        S.cmaStart = { x: nx, y: ny };
        rebuild(); // deterministic restart from the new start (same seed)
      },
      onEnd: function () { draw(); }
    });

    // ---- Selection wiring ------------------------------------------------
    function syncProblemOptions() {
      if (S.algo === "nsga") {
        probSel.setOptions(MO_PROBLEMS);
        probSel.set(S.mo);
      } else {
        probSel.setOptions(LANDSCAPES);
        probSel.set(S.landscape);
      }
    }

    algoSel.onChange(function (idx) {
      anim.pause();
      S.algo = ALGOS[idx].key;
      syncProblemOptions();
      buildAlgoControls();
      buildReadouts();
      rebuild();
    });

    probSel.onChange(function () {
      var v = probSel.value();
      if (S.algo === "nsga") {
        S.mo = v;
      } else {
        S.landscape = v;
        S.cmaStart = null; // new bounds -> new default start
      }
      rebuild();
    });

    FV.onThemeChange(function () {
      heatKey = ""; // heat image bakes the ink color — rebuild it
      draw();
    });

    function algoLabel() {
      for (var i = 0; i < ALGOS.length; i++) {
        if (ALGOS[i].key === S.algo) return ALGOS[i].label;
      }
      return S.algo;
    }

    // ---- boot ------------------------------------------------------------
    if (anim.reduced) {
      btns.fvButtons["Run"].title = "Reduced motion is on — Run advances " + BATCH + " generations at once";
    }
    algoSel.set(ALGOS[0].label);
    syncProblemOptions();
    buildAlgoControls();
    buildReadouts();
    rebuild();
  }

  // ==== small helpers =======================================================
  function mk(tag, cls, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  }

  function clampNum(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function fmtInt(v) {
    return String(Math.round(v));
  }

  function fmt2(v) {
    return (+v).toFixed(2);
  }

  function fmtF(v) {
    if (!isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(2);
    return v.toFixed(3);
  }

  function safeVersion(W) {
    try { return W.version(); } catch (e) { return ""; }
  }

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

  function dot(ctx, x, y, r, color, alpha) {
    if (!isFinite(x) || !isFinite(y)) return;
    ctx.save();
    ctx.globalAlpha = alpha != null ? alpha : 1;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.restore();
  }

  function ring(ctx, x, y, r, color, alpha, width) {
    if (!isFinite(x) || !isFinite(y)) return;
    ctx.save();
    ctx.globalAlpha = alpha != null ? alpha : 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 1.5;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
  }

  function cross(ctx, x, y, r, color, alpha) {
    if (!isFinite(x) || !isFinite(y)) return;
    ctx.save();
    ctx.globalAlpha = alpha != null ? alpha : 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x - r, y);
    ctx.lineTo(x + r, y);
    ctx.moveTo(x, y - r);
    ctx.lineTo(x, y + r);
    ctx.stroke();
    ctx.restore();
  }

  function bestMarker(ctx, x, y, C) {
    dot(ctx, x, y, 3, C.hot, 1);
    ring(ctx, x, y, 6.5, C.hot, 0.85, 1.5);
  }

  // Parametric ellipse in DATA coords mapped through the scales (the x/y
  // pixel densities differ, so ctx.ellipse would draw the wrong shape).
  function ellipsePts(sx, sy, cx, cy, r1, r2, angle) {
    var pts = [], n = 48;
    var ca = Math.cos(angle), sa = Math.sin(angle);
    for (var i = 0; i <= n; i++) {
      var t = (i / n) * 2 * Math.PI;
      var ex = r1 * Math.cos(t), ey = r2 * Math.sin(t);
      pts.push([sx(cx + ca * ex - sa * ey), sy(cy + sa * ex + ca * ey)]);
    }
    return pts;
  }

  // Analytic Pareto reference fronts (references only — no algorithm math).
  function refFrontPts(prob) {
    var pts = [], i;
    if (prob === "schaffer") {
      // Parametric: x = t in [0, 2] -> (t², (t−2)²).
      for (i = 0; i <= 160; i++) {
        var t = 2 * i / 160;
        pts.push([t * t, (t - 2) * (t - 2)]);
      }
      return pts;
    }
    var n = prob === "zdt3" ? 400 : 160;
    for (i = 0; i <= n; i++) {
      var f1 = i / n, f2;
      if (prob === "zdt1") f2 = 1 - Math.sqrt(f1);
      else if (prob === "zdt2") f2 = 1 - f1 * f1;
      else f2 = 1 - Math.sqrt(f1) - f1 * Math.sin(10 * Math.PI * f1);
      pts.push([f1, f2]);
    }
    return pts;
  }

  function cvLabel(cv, th, txt) {
    var ctx = cv.ctx;
    ctx.save();
    ctx.fillStyle = th.colors.ink;
    ctx.globalAlpha = 0.75;
    ctx.font = "11px var(--mono-font, monospace)";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(txt, cv.w - 12, 4);
    ctx.restore();
  }

  // Native <select> wrapped in the fv-control shell so it themes with the
  // sliders. Returns {el, value(), set(v), onChange(fn), setOptions(list),
  // index()}.
  function selectControl(parent, labelTxt, options) {
    var wrap = mk("label", "fv-control", parent);
    var lab = mk("span", "fv-control-label", wrap);
    lab.textContent = labelTxt;
    var sel = mk("select", "fv-select", wrap);
    var handler = null;
    fill(options);
    function fill(list) {
      sel.innerHTML = "";
      for (var i = 0; i < list.length; i++) {
        var o = document.createElement("option");
        o.value = list[i];
        o.textContent = list[i];
        sel.appendChild(o);
      }
    }
    sel.addEventListener("change", function () {
      if (handler) handler(sel.selectedIndex);
    });
    return {
      el: wrap,
      value: function () { return sel.value; },
      index: function () { return sel.selectedIndex < 0 ? 0 : sel.selectedIndex; },
      set: function (v) { sel.value = v; },
      setOptions: fill,
      onChange: function (fn) { handler = fn; }
    };
  }

  // A Victor-style scrubbable number inside the fv-control shell (label above,
  // draggable mono value below). Returns the scrub span (has fvGet/fvSet).
  function scrubControl(parent, labelTxt, o) {
    var wrap = mk("label", "fv-control", parent);
    var lab = mk("span", "fv-control-label", wrap);
    lab.textContent = labelTxt;
    var span = mk("span", "", wrap);
    return FV.scrub(span, o);
  }
})();
