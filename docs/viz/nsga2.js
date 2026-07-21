// docs/viz/nsga2.js — "A Pareto Front Forming Live"
// NSGA-II on a two-objective benchmark (zdt1|zdt2|zdt3|schaffer), rendered as
// an objective-space scatter: the population starts scattered high in (f1, f2)
// space and converges DOWN onto the analytic reference front (both objectives
// minimized — lower-left is better).
//   yellow (data): the analytic reference front, a fixed target curve.
//   green  (post): rank-0 individuals — the evolved non-dominated front,
//                  connected by a thin polyline sorted by f1.
//   faded          dominated individuals, fading toward the grid color as
//                  their Pareto rank grows.
//   coral  (hot):  small halos on rank-0 boundary points (infinite crowding
//                  distance, encoded as crowding = -1 by the wasm engine).
// WASM-ONLY compute: every generation is stepped by the REAL fugue-evo crate
// via crates/fugue-evo-wasm (ExploreNsga2). No algorithm math is mirrored in
// JS — only the analytic reference curves (targets, not algorithm state) are
// evaluated here for drawing. Self-contained IIFE; requires fugue-viz.js and
// fugue-evo-wasm-loader.js to have loaded first.
(function () {
  "use strict";
  if (typeof window === "undefined" || !window.FugueViz) return;
  var FV = window.FugueViz;

  var PROBLEMS = ["zdt1", "zdt2", "zdt3", "schaffer"];
  var PACE_HZ = 4;        // generations per second while playing
  var REDUCED_GENS = 40;  // synchronous generations under prefers-reduced-motion

  // ---- color helpers (fade dominated points toward the grid color) ---------
  function toRgba(col) {
    col = (col || "").trim();
    var m = /^#?([0-9a-f]{6})$/i.exec(col);
    if (m) {
      var n = parseInt(m[1], 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255, 1];
    }
    var r = /rgba?\(([^)]+)\)/.exec(col);
    if (r) {
      var p = r[1].split(",");
      return [
        parseInt(p[0], 10), parseInt(p[1], 10), parseInt(p[2], 10),
        p.length > 3 ? parseFloat(p[3]) : 1
      ];
    }
    return [128, 128, 128, 1];
  }
  function mixRgba(a, b, t) {
    return "rgba(" +
      Math.round(a[0] + (b[0] - a[0]) * t) + "," +
      Math.round(a[1] + (b[1] - a[1]) * t) + "," +
      Math.round(a[2] + (b[2] - a[2]) * t) + "," +
      Math.max(0.12, a[3] + (b[3] - a[3]) * t).toFixed(3) + ")";
  }

  // ---- analytic reference fronts (target curves, drawn faint in data color).
  // These are the textbook closed forms of each benchmark's true front — a
  // drawing reference, NOT a re-implementation of NSGA-II. For zdt3 the full
  // curve is drawn; its lower envelope is the true (disconnected) front.
  function refPoints(problem) {
    var pts = [], i, N;
    if (problem === "schaffer") {
      N = 160;
      for (i = 0; i <= N; i++) {
        var t = 2 * i / N;
        pts.push([t * t, (t - 2) * (t - 2)]);
      }
      return pts;
    }
    N = 240;
    for (i = 0; i <= N; i++) {
      var f1 = i / N;
      var f2;
      if (problem === "zdt2") f2 = 1 - f1 * f1;
      else if (problem === "zdt3") f2 = 1 - Math.sqrt(f1) - f1 * Math.sin(10 * Math.PI * f1);
      else f2 = 1 - Math.sqrt(f1); // zdt1
      pts.push([f1, f2]);
    }
    return pts;
  }

  // Init defers on FV.wasmReady (fugue-evo-wasm-loader.js): resolves to the
  // wasm-bindgen module namespace, or null when the pkg is absent. This widget
  // is WASM-ONLY — without the real crate it shows a notice, never fake math.
  FV.register("nsga2-front", function (root, FV) {
    var p = FV.wasmReady || Promise.resolve(null);
    p.then(function (W) {
      if (!W) {
        root.setAttribute("data-fugue-backend", "none");
        var notice = document.createElement("div");
        notice.className = "fv-pg-notice";
        notice.textContent = "This figure runs the real fugue-evo crate compiled to WebAssembly — the wasm package isn't available in this build.";
        root.appendChild(notice);
        return;
      }
      root.setAttribute("data-fugue-backend", "wasm");
      realInit(root, FV, W);
    });
  });

  function realInit(root, FV, W) {
    // ------------------------------------------------------------------ state
    var seed0 = parseInt(root.getAttribute("data-seed") || "11", 10);
    var prob0 = root.getAttribute("data-problem") || "zdt1";
    var pop0 = parseInt(root.getAttribute("data-pop") || "60", 10);
    if (!isFinite(seed0) || seed0 < 1) seed0 = 11;
    if (PROBLEMS.indexOf(prob0) < 0) prob0 = "zdt1";
    if (!isFinite(pop0)) pop0 = 60;
    if (pop0 < 8) pop0 = 8;       // mirror the engine's clamp so the
    if (pop0 > 256) pop0 = 256;   // "front N/POP" readout denominator is honest

    var params = { seed: seed0, problem: prob0, pop: pop0 };
    var eng = null;      // ExploreNsga2 instance (rebuilt on seed/problem/reset)
    var snap = null;     // parsed payload {gen, points:[[f1,f2,rank,crowding]...], fronts}
    var domX = [0, 1.05], domY = [0, 4.5];  // fixed after each rebuild
    var refPix = null;   // reference-front data coords for the current problem

    function rebuild() {
      eng = null;
      snap = null;
      try {
        // popSize is clamped to [8,256] inside the engine; seed is u64 -> BigInt.
        eng = new W.ExploreNsga2(params.problem, params.pop, BigInt(params.seed));
      } catch (e) {
        try { console.error("[fugue-viz] nsga2: engine construction failed", e); } catch (e2) { /* readout only */ }
        return;
      }
      // snapshot() paints the initial (unranked, rank=-1) population without
      // advancing — the same deterministic start every time for this seed.
      snap = JSON.parse(eng.snapshot());
      fitAxes();
      refPix = refPoints(params.problem);
      // Under reduced motion there is no animation: converge synchronously so
      // the single static frame shows a formed front, never a random cloud.
      if (loopApi && loopApi.reduced) advance(REDUCED_GENS);
    }

    // Axes are fixed after init so convergence reads as downward motion onto
    // the reference curve (a rescaling axis would hide the motion). For zdt*
    // the f2 top fits the initial scatter ONCE (min top 4.5); zdt3's true
    // front dips below zero, so its floor is -1. Schaffer is fixed 20x20.
    function fitAxes() {
      if (params.problem === "schaffer") {
        domX = [0, 20];
        domY = [0, 20];
        return;
      }
      var top = 0;
      if (snap && snap.points) {
        for (var i = 0; i < snap.points.length; i++) {
          var f2 = snap.points[i][1];
          if (isFinite(f2) && f2 > top) top = f2;
        }
      }
      domX = [0, 1.05];
      domY = [params.problem === "zdt3" ? -1 : 0, Math.max(4.5, top * 1.05)];
    }

    function advance(n) {
      if (!eng) return;
      for (var i = 0; i < n; i++) snap = JSON.parse(eng.step());
    }

    // --------------------------------------------------------------- DOM shell
    var controls = document.createElement("div");
    controls.className = "fv-controls";
    root.appendChild(controls);

    // problem chooser
    var probWrap = document.createElement("label");
    probWrap.className = "fv-control";
    var probLab = document.createElement("span");
    probLab.className = "fv-control-label";
    probLab.textContent = "PROBLEM";
    probWrap.appendChild(probLab);
    var probSel = document.createElement("select");
    probSel.className = "fv-select";
    for (var pi = 0; pi < PROBLEMS.length; pi++) {
      var opt = document.createElement("option");
      opt.value = PROBLEMS[pi];
      opt.textContent = PROBLEMS[pi];
      probSel.appendChild(opt);
    }
    probSel.value = params.problem;
    probSel.addEventListener("change", function () {
      params.problem = probSel.value;
      rebuild();
      renderReadouts();
      requestDraw();
    });
    probWrap.appendChild(probSel);
    controls.appendChild(probWrap);

    var btns = FV.buttons(controls, [
      { label: "Play", title: "Run generations (~4/s)", primary: true, onClick: function () { togglePlay(); } },
      { label: "Step", title: "Advance one generation", onClick: function () { loopApi.step(); } },
      { label: "Reset", title: "Restart from the seeded initial population", onClick: function () { rebuild(); renderReadouts(); requestDraw(); } }
    ]);

    // seed scrub (Victor-style draggable number) — reseeding rebuilds the
    // engine deterministically: same seed, same run, every time.
    var seedWrap = document.createElement("label");
    seedWrap.className = "fv-control";
    var seedLab = document.createElement("span");
    seedLab.className = "fv-control-label";
    seedLab.textContent = "SEED";
    seedWrap.appendChild(seedLab);
    var seedSpan = document.createElement("span");
    seedWrap.appendChild(seedSpan);
    controls.appendChild(seedWrap);
    FV.scrub(seedSpan, {
      min: 1, max: 999, step: 1, value: params.seed,
      onInput: function (v) {
        params.seed = v | 0;
        rebuild();
        renderReadouts();
        requestDraw();
      }
    });

    var cv = FV.canvas(root, { height: 340, onResize: function () { requestDraw(); } });
    var ctx = cv.ctx;

    var readouts = document.createElement("div");
    readouts.className = "fv-readouts";
    root.appendChild(readouts);
    var rGen = FV.readout(readouts, { label: "GEN" });
    var rFront = FV.readout(readouts, { label: "FRONT" });

    var hint = document.createElement("div");
    hint.className = "fv-hint";
    hint.textContent = "both objectives are minimized — watch the green rank-0 front push down-left onto the yellow reference curve; scrub the seed and a different random cloud converges to the same front.";
    root.appendChild(hint);

    function renderReadouts() {
      if (!snap) {
        rGen.set("—");
        rFront.set("—");
        return;
      }
      var total = snap.points.length, n0 = 0;
      for (var i = 0; i < total; i++) {
        if (snap.points[i][2] === 0) n0++;
      }
      rGen.set(String(snap.gen));
      rFront.set(n0 + "/" + total, n0 === total ? "post" : null);
    }

    // ------------------------------------------------------------------- draw
    function draw() {
      if (!cv) return; // onResize can fire during canvas() before cv is assigned
      var w = cv.w, h = cv.h;
      cv.clear();
      var th = FV.theme(), C = th.colors;

      var pad = { l: 46, r: 12, t: 12, b: 30 };
      var iw = Math.max(10, w - pad.l - pad.r);
      var ih = Math.max(10, h - pad.t - pad.b);
      var ix = pad.l, iy = pad.t;
      var sx = FV.scale(domX, [ix, ix + iw]);
      var sy = FV.scale(domY, [iy + ih, iy]);

      FV.axes(ctx, { x: ix, y: iy, w: iw, h: ih, xscale: sx, yscale: sy, xlabel: "f1", ylabel: "f2", theme: th });

      if (!snap) return;

      // Everything data-driven clips to the plot frame: early generations can
      // carry huge objective values far outside the fixed window.
      ctx.save();
      ctx.beginPath();
      ctx.rect(ix, iy, iw, ih);
      ctx.clip();

      // analytic reference front — faint yellow target curve
      var i, refP = new Array(refPix.length);
      for (i = 0; i < refPix.length; i++) refP[i] = [sx(refPix[i][0]), sy(refPix[i][1])];
      ctx.save();
      ctx.globalAlpha = 0.45;
      FV.curve(ctx, refP, { color: C.data, width: 1.5 });
      ctx.restore();

      var postRgba = toRgba(C.post);
      var gridRgba = toRgba(C.grid);
      var pts = snap.points;
      var rank0 = [];

      // dominated points first (under the front), fading toward grid with rank;
      // rank -1 (initial unranked population) draws mid-fade.
      for (i = 0; i < pts.length; i++) {
        var p = pts[i];
        if (!isFinite(p[0]) || !isFinite(p[1])) continue;
        if (p[2] === 0) { rank0.push(p); continue; }
        var t = p[2] < 0 ? 0.55 : Math.min(1, p[2] / 4);
        ctx.fillStyle = mixRgba(postRgba, gridRgba, t);
        ctx.beginPath();
        ctx.arc(sx(p[0]), sy(p[1]), 2.5, 0, 2 * Math.PI);
        ctx.fill();
      }

      // rank-0: thin green polyline sorted by f1 — the evolved front...
      rank0.sort(function (a, b) { return a[0] - b[0]; });
      var frontP = new Array(rank0.length);
      for (i = 0; i < rank0.length; i++) frontP[i] = [sx(rank0[i][0]), sy(rank0[i][1])];
      ctx.save();
      ctx.globalAlpha = 0.55;
      FV.curve(ctx, frontP, { color: C.post, width: 1.2 });
      ctx.restore();

      // ...with full-alpha green points, coral halos on the boundary points
      // (crowding = -1 encodes an infinite crowding distance).
      for (i = 0; i < rank0.length; i++) {
        var q = rank0[i];
        var px = sx(q[0]), py = sy(q[1]);
        if (q[3] === -1) FV.halo(ctx, px, py, 6.5, C.hot, 0.3);
        ctx.fillStyle = C.post;
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, 2 * Math.PI);
        ctx.fill();
      }

      ctx.restore(); // end plot clip
    }

    // schedule a single draw when paused (during play the loop already draws)
    var drawQueued = false;
    function requestDraw() {
      if (loopApi && loopApi.playing) return;
      if (drawQueued) return;
      drawQueued = true;
      window.requestAnimationFrame(function () { drawQueued = false; draw(); });
    }

    // ------------------------------------------------------------------- loop
    // ~4 generations/sec regardless of frame rate (FV.pace carries the
    // sub-tick remainder); Step advances exactly one generation.
    var pacer = FV.pace(PACE_HZ);
    var loopApi = FV.loop(root, function (dt) {
      if (!eng) return;
      if (dt === 0) advance(1);                 // Step button / reduced-motion
      else advance(pacer(dt));
      renderReadouts();
      draw();
    }, { autoplay: true });

    function togglePlay() {
      if (loopApi.playing) { loopApi.pause(); btns.fvButtons["Play"].textContent = "Play"; }
      else { loopApi.play(); if (loopApi.playing) btns.fvButtons["Play"].textContent = "Pause"; }
    }

    if (loopApi.reduced) {
      btns.fvButtons["Play"].textContent = "Play";
      btns.fvButtons["Play"].disabled = true;
      btns.fvButtons["Play"].title = "Reduced motion is on — use Step";
      btns.fvButtons["Step"].classList.add("fv-primary");
    }

    FV.onThemeChange(function () { requestDraw(); });

    // first paint: the seeded initial population (or, under reduced motion,
    // the synchronously converged front from rebuild()).
    rebuild();
    renderReadouts();
    draw();
    // The loop autoplays itself (see FV.loop {autoplay:true}); reflect that on
    // the button. play() already ran and is a no-op under reduced motion.
    if (loopApi.playing) btns.fvButtons["Play"].textContent = "Pause";
  }
})();
