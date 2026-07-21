/* fugue-evo-wasm loader — makes the REAL fugue-evo crate available to the
 * doc widgets and the playground.
 *
 * Publishes `FV.wasmReady`, a promise resolving to the wasm-bindgen module
 * namespace (ExploreGa, ExploreCma, ExploreNsga2, ExploreIsland,
 * ExploreUmda, explore_landscape_grid, ...) once
 * `<site>/pkg/fugue_evo_wasm.js` has loaded and initialized — or to `null`
 * when the package is absent (local builds without wasm-pack) or the
 * browser cannot load it. Widgets await this promise at init time; without
 * the module they show a short notice instead of animating, so the docs
 * never silently present imitation math as the crate.
 *
 * Must be listed in book.toml after fugue-viz.js and before viz/*.js.
 */
(function () {
  'use strict';
  if (!window.FugueViz) return;
  var FV = window.FugueViz;

  var script =
    document.currentScript ||
    document.querySelector('script[src*="fugue-evo-wasm-loader"]');
  var root = script ? script.src.replace(/fugue-evo-wasm-loader\.js.*$/, '') : './';

  // Dynamic import via Function so browsers without import() support fail
  // at call time (caught below), not while parsing this whole file.
  var dynImport;
  try {
    dynImport = new Function('u', 'return import(u)');
  } catch (e) {
    FV.wasm = null;
    FV.wasmReady = Promise.resolve(null);
    return;
  }

  FV.wasm = null;
  FV.wasmReady = dynImport(root + 'pkg/fugue_evo_wasm.js')
    .then(function (mod) {
      return mod.default({ module_or_path: root + 'pkg/fugue_evo_wasm_bg.wasm' }).then(function () {
        FV.wasm = mod;
        try {
          console.log(
            '[fugue-viz] fugue-evo-wasm ' +
              mod.version() +
              ' loaded — widgets run the real crate'
          );
        } catch (e) { /* readout only */ }
        return mod;
      });
    })
    .catch(function (e) {
      try {
        console.log(
          '[fugue-viz] fugue-evo-wasm unavailable (' +
            (e && e.message ? e.message : e) +
            ') — widgets are disabled'
        );
      } catch (e2) { /* readout only */ }
      return null;
    });
})();
