/* Evolution-as-inference explorable: tempered SMC over the Boltzmann
 * posterior, run by the REAL fugue-evo inference layer compiled to wasm
 * (ExploreSmcInference). One tempering rung per tick: the heat is the exact
 * tempered density -(log p + beta*f) recomputed from the crate each rung, so
 * the particle cloud can be watched matching the analytic target as beta
 * climbs from the prior (beta = 0) to the posterior (beta = 1) and onward
 * into annealed-optimizer territory. No JS fallback math. */
(function () {
  "use strict";
  if (!window.FugueViz) return;
  var FV = window.FugueViz;

  var NOTICE =
    "This figure runs the real fugue-evo crate compiled to WebAssembly — the wasm package isn't available in this build.";
  var HEAT_N = 110;

  /* ---- local helpers (per-file convention) ---- */
  function el(tag, cls, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  }
  function showNotice(root, msg) {
    var d = el("div", "fv-pg-notice", root);
    d.textContent = msg;
  }
  function mkPlot(w, h, dom) {
    var pad = { l: 40, r: 12, t: 10, b: 26 };
    var ix = pad.l,
      iy = pad.t,
      iw = Math.max(10, w - pad.l - pad.r),
      ih = Math.max(10, h - pad.t - pad.b);
    return {
      ix: ix,
      iy: iy,
      iw: iw,
      ih: ih,
      sx: FV.scale(dom, [ix, ix + iw]),
      sy: FV.scale(dom, [iy + ih, iy]),
    };
  }
  function dot(ctx, x, y, r, color, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
  function diamond(ctx, x, y, s, color) {
    ctx.save();
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    ctx.moveTo(x, y - s);
    ctx.lineTo(x + s, y);
    ctx.lineTo(x, y + s);
    ctx.lineTo(x - s, y);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
  function fmtF(v) {
    if (!isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a >= 1e4 || (a < 1e-3 && a > 0)) return v.toExponential(2);
    return v.toFixed(3);
  }
  function buildHeat(off, grid, nx, ny, colHex) {
    off.width = nx;
    off.height = ny;
    var octx = off.getContext("2d");
    var img = octx.createImageData(nx, ny);
    var mn = Infinity,
      mx = -Infinity;
    for (var k = 0; k < grid.length; k++) {
      if (grid[k] < mn) mn = grid[k];
      if (grid[k] > mx) mx = grid[k];
    }
    var span = mx - mn || 1;
    var col = hex(colHex);
    for (var j = 0; j < ny; j++) {
      for (var i = 0; i < nx; i++) {
        var t = (grid[j * nx + i] - mn) / span;
        var a = 0.6 * (1 - Math.sqrt(t));
        var p = ((ny - 1 - j) * nx + i) * 4;
        img.data[p] = col[0];
        img.data[p + 1] = col[1];
        img.data[p + 2] = col[2];
        img.data[p + 3] = Math.round(255 * a);
      }
    }
    octx.putImageData(img, 0, 0);
  }
  function hex(h) {
    h = (h || "#f2cc60").replace("#", "");
    if (h.length === 3)
      h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    return [
      parseInt(h.slice(0, 2), 16) || 242,
      parseInt(h.slice(2, 4), 16) || 204,
      parseInt(h.slice(4, 6), 16) || 96,
    ];
  }
  function seedControl(controls, value, onInput) {
    var wrap = el("div", "fv-control", controls);
    var lab = el("span", "fv-control-label", wrap);
    lab.textContent = "SEED";
    var span = el("span", "fv-control-value", wrap);
    FV.scrub(span, { min: 1, max: 999, step: 1, value: value, onInput: onInput });
  }

  /* ---- widget ---- */
  function smcInit(root, W) {
    var params = {
      seed: parseInt(root.getAttribute("data-seed") || "11", 10) || 11,
      pop: parseInt(root.getAttribute("data-pop") || "240", 10) || 240,
      rungs: parseInt(root.getAttribute("data-rungs") || "18", 10) || 18,
      betamax: parseFloat(root.getAttribute("data-betamax") || "1") || 1,
      crossover: (root.getAttribute("data-crossover") || "on") !== "off",
    };

    var engine = null,
      cur = null,
      info = null,
      DOM = [-4.5, 4.5];
    var heatCv = document.createElement("canvas");
    var heatDirty = true;

    function advance() {
      cur = JSON.parse(engine.step());
      heatDirty = true; // the tempered density changes with beta
    }
    function rebuild() {
      engine = new W.ExploreSmcInference(
        params.pop,
        params.rungs,
        params.betamax,
        params.crossover,
        BigInt(params.seed)
      );
      info = JSON.parse(engine.info());
      DOM = [info.lo, info.hi];
      cur = JSON.parse(engine.snapshot());
      heatDirty = true;
      if (loopApi && loopApi.reduced) {
        // Reduced motion: run the whole ladder synchronously, show the end.
        while (!cur.done) advance();
      }
    }

    /* controls -> canvas -> instruction -> readouts */
    var controls = el("div", "fv-controls", root);
    FV.slider(controls, {
      label: "PARTICLES",
      min: 48,
      max: 480,
      step: 16,
      value: params.pop,
      fmt: function (v) {
        return String(Math.round(v));
      },
      onInput: function (v) {
        params.pop = Math.round(v);
        rebuild();
        renderReadouts();
        requestDraw();
      },
    });
    FV.slider(controls, {
      label: "RUNGS",
      min: 6,
      max: 40,
      step: 1,
      value: params.rungs,
      fmt: function (v) {
        return String(Math.round(v));
      },
      onInput: function (v) {
        params.rungs = Math.round(v);
        rebuild();
        renderReadouts();
        requestDraw();
      },
    });
    FV.slider(controls, {
      label: "β MAX",
      min: 1,
      max: 6,
      step: 0.5,
      value: params.betamax,
      fmt: function (v) {
        return "×" + v.toFixed(1);
      },
      onInput: function (v) {
        params.betamax = v;
        rebuild();
        renderReadouts();
        requestDraw();
      },
    });
    FV.toggle(controls, {
      label: "CROSSOVER",
      value: params.crossover,
      onChange: function (v) {
        params.crossover = v;
        rebuild();
        renderReadouts();
        requestDraw();
      },
    });
    seedControl(controls, params.seed, function (v) {
      params.seed = v | 0;
      rebuild();
      renderReadouts();
      requestDraw();
    });
    var btns = FV.buttons(controls, [
      {
        label: "Play",
        title: "Run the tempering ladder",
        primary: true,
        onClick: togglePlay,
      },
      {
        label: "Step",
        title: "Advance one tempering rung",
        onClick: function () {
          loopApi.step();
        },
      },
      {
        label: "Reset",
        title: "Rebuild from the current seed",
        onClick: function () {
          rebuild();
          renderReadouts();
          requestDraw();
          if (!loopApi.playing) setPlayLabel("Play");
        },
      },
    ]);

    var cv = FV.canvas(root, {
      height: 300,
      onResize: function () {
        draw();
      },
    });
    var ctx = cv.ctx;

    var instr = el("div", "fv-instruction", root);
    instr.textContent =
      "yellow heat = the exact tempered target πβ ∝ p(x)·exp(β·f(x)) · green dots = SMC particles (size = weight) · blue rings = the prior's 1σ/2σ · coral diamonds = the two fitness peaks";

    var readouts = el("div", "fv-readouts", root);
    var rBeta = FV.readout(readouts, { label: "β" });
    var rRung = FV.readout(readouts, { label: "RUNG" });
    var rEss = FV.readout(readouts, { label: "ESS" });
    var rZ = FV.readout(readouts, { label: "log Z" });
    var rSwap = FV.readout(readouts, { label: "SWAPS" });

    function renderReadouts() {
      if (!cur) return;
      rBeta.set(cur.beta.toFixed(2), "post");
      rRung.set(cur.rung + "/" + cur.n_rungs);
      rEss.set(String(Math.round(cur.ess)), cur.resampled ? "hot" : "post");
      rZ.set(fmtF(cur.log_evidence), "data");
      rSwap.set(params.crossover ? String(cur.swaps) : "off", "flow");
    }

    function draw() {
      if (!cv) return;
      cv.clear();
      var th = FV.theme(),
        C = th.colors;
      var plot = mkPlot(cv.w, cv.h, DOM);
      if (heatDirty && engine) {
        buildHeat(heatCv, engine.density_grid(HEAT_N, HEAT_N), HEAT_N, HEAT_N, C.data);
        heatDirty = false;
      }
      ctx.drawImage(heatCv, plot.ix, plot.iy, plot.iw, plot.ih);
      FV.axes(ctx, {
        x: plot.ix,
        y: plot.iy,
        w: plot.iw,
        h: plot.ih,
        xscale: plot.sx,
        yscale: plot.sy,
        xlabel: "x",
        ylabel: "y",
        theme: th,
      });
      if (!cur || !info) return;
      ctx.save();
      ctx.beginPath();
      ctx.rect(plot.ix, plot.iy, plot.iw, plot.ih);
      ctx.clip();

      // Prior 1-sigma / 2-sigma rings (the beta = 0 starting law).
      var cx = plot.sx(0),
        cyp = plot.sy(0);
      var r1 = Math.abs(plot.sx(info.prior_std) - plot.sx(0));
      ctx.save();
      ctx.strokeStyle = C.prior;
      ctx.globalAlpha = 0.5;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(cx, cyp, r1, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([4, 4]);
      ctx.globalAlpha = 0.3;
      ctx.beginPath();
      ctx.arc(cx, cyp, 2 * r1, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // Fitness peaks.
      for (var m = 0; m < info.modes.length; m++) {
        diamond(ctx, plot.sx(info.modes[m][0]), plot.sy(info.modes[m][1]), 5, C.hot);
      }

      // Particles: radius and alpha carry the normalized weight.
      var n = cur.particles.length;
      for (var i = 0; i < n; i++) {
        var p = cur.particles[i];
        var w = p[2];
        var r = Math.min(6, 1.6 + Math.sqrt(Math.max(0, w) * n) * 1.6);
        var a = Math.min(0.9, 0.25 + w * n * 0.35);
        dot(ctx, plot.sx(p[0]), plot.sy(p[1]), r, C.post, a);
      }
      ctx.restore();
    }

    var drawQueued = false;
    function requestDraw() {
      if (loopApi.playing || drawQueued) return;
      drawQueued = true;
      window.requestAnimationFrame(function () {
        drawQueued = false;
        draw();
      });
    }

    var pacer = FV.pace(2.5);
    var loopApi = FV.loop(
      root,
      function (dt) {
        if (!engine) return;
        if (dt === 0) {
          if (!cur.done) advance();
          renderReadouts();
          draw();
          return;
        }
        if (cur.done) {
          loopApi.pause();
          setPlayLabel("Replay");
          draw();
          return;
        }
        var ticks = pacer(dt);
        while (ticks-- > 0 && !cur.done) advance();
        renderReadouts();
        draw();
      },
      { autoplay: true }
    );

    function setPlayLabel(txt) {
      btns.fvButtons["Play"].textContent = txt;
    }
    function togglePlay() {
      if (loopApi.playing) {
        loopApi.pause();
        setPlayLabel("Play");
      } else {
        if (cur && cur.done) {
          rebuild();
          renderReadouts();
        }
        loopApi.play();
        if (loopApi.playing) setPlayLabel("Pause");
      }
    }
    if (loopApi.reduced) {
      btns.fvButtons["Play"].disabled = true;
      btns.fvButtons["Play"].title =
        "Reduced motion is on — the completed ladder is shown";
      btns.fvButtons["Step"].disabled = true;
    }

    FV.onThemeChange(function () {
      heatDirty = true;
      draw();
    });

    rebuild();
    renderReadouts();
    draw();
    if (loopApi.playing) setPlayLabel("Pause");
  }

  FV.register("smc-inference", function (root) {
    var p = FV.wasmReady || Promise.resolve(null);
    p.then(function (W) {
      if (!W) {
        root.setAttribute("data-fugue-backend", "none");
        showNotice(root, NOTICE);
        return;
      }
      root.setAttribute("data-fugue-backend", "wasm");
      try {
        smcInit(root, W);
      } catch (e) {
        try {
          console.error("[fugue-viz] smc-inference init failed", e);
        } catch (e2) {}
        showNotice(
          root,
          "smc-inference failed to initialize: " + (e && e.message ? e.message : e)
        );
      }
    });
  });
})();
