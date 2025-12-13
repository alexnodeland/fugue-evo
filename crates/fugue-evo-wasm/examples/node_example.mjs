/**
 * Node.js example for fugue-evo-wasm
 *
 * Run with:
 *   cd crates/fugue-evo-wasm
 *   wasm-pack build --target nodejs
 *   node examples/node_example.mjs
 */

import { createRequire } from 'module';
const require = createRequire(import.meta.url);

// Load the WASM module (Node.js target)
const wasm = require('../pkg/fugue_evo_wasm.js');

console.log('='.repeat(60));
console.log('fugue-evo-wasm Node.js Example');
console.log(`Version: ${wasm.version()}`);
console.log('='.repeat(60));

// ============================================================================
// Real Vector Optimization
// ============================================================================

console.log('\n--- Real Vector Optimization (Sphere Function) ---');

const rvOptimizer = new wasm.RealVectorOptimizer(10);
rvOptimizer.setPopulationSize(100);
rvOptimizer.setMaxGenerations(200);
rvOptimizer.setBounds(-5.12, 5.12);
rvOptimizer.setFitness('sphere');
rvOptimizer.setSeed(42);

const rvResult = rvOptimizer.optimize();
console.log(`Best fitness: ${rvResult.bestFitness.toExponential(4)}`);
console.log(`Generations: ${rvResult.generations}`);
console.log(`Evaluations: ${rvResult.evaluations}`);
console.log(`Best solution (first 5 dims): [${rvResult.bestGenome.slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
rvOptimizer.free();

// ============================================================================
// Custom Fitness Function
// ============================================================================

console.log('\n--- Custom Fitness Function ---');

const customOptimizer = new wasm.RealVectorOptimizer(5);
customOptimizer.setPopulationSize(50);
customOptimizer.setMaxGenerations(100);
customOptimizer.setBounds(-10, 10);
customOptimizer.setSeed(123);

// Custom fitness: maximize the sum (move towards upper bound)
const customResult = customOptimizer.optimizeCustom((genome) => {
    return genome.reduce((sum, x) => sum + x, 0);
});

console.log(`Best fitness (sum): ${customResult.bestFitness.toFixed(4)}`);
console.log(`Best solution: [${customResult.bestGenome.map(x => x.toFixed(4)).join(', ')}]`);
customOptimizer.free();

// ============================================================================
// CMA-ES Optimization
// ============================================================================

console.log('\n--- CMA-ES Optimization (Rosenbrock) ---');

const cmaOptimizer = new wasm.RealVectorOptimizer(5);
cmaOptimizer.setAlgorithm(wasm.Algorithm.CmaES);
cmaOptimizer.setCmaEsSigma(1.0);
cmaOptimizer.setMaxGenerations(300);
cmaOptimizer.setBounds(-5, 5);
cmaOptimizer.setFitness('rosenbrock');
cmaOptimizer.setSeed(42);

const cmaResult = cmaOptimizer.optimize();
console.log(`Best fitness: ${cmaResult.bestFitness.toExponential(4)}`);
console.log(`Solution: [${cmaResult.bestGenome.map(x => x.toFixed(4)).join(', ')}]`);
cmaOptimizer.free();

// ============================================================================
// BitString Optimization (OneMax)
// ============================================================================

console.log('\n--- BitString Optimization (OneMax) ---');

const bitOptimizer = new wasm.BitStringOptimizer(50);
bitOptimizer.setPopulationSize(100);
bitOptimizer.setMaxGenerations(100);
bitOptimizer.setFlipProbability(0.02);
bitOptimizer.setSeed(42);

const bitResult = bitOptimizer.solveOneMax();
console.log(`Best fitness (ones count): ${bitResult.bestFitness}`);
console.log(`Best genome (string): ${bitResult.bestGenomeString()}`);
console.log(`Count of ones: ${bitResult.countOnes()}`);
bitOptimizer.free();

// ============================================================================
// Permutation Optimization (Simple Distance)
// ============================================================================

console.log('\n--- Permutation Optimization ---');

const permOptimizer = new wasm.PermutationOptimizer(8);
permOptimizer.setPopulationSize(50);
permOptimizer.setMaxGenerations(100);
permOptimizer.setSeed(42);

// Fitness: minimize displacement from sorted order (maximize negative displacement)
const permResult = permOptimizer.optimize((perm) => {
    let displacement = 0;
    for (let i = 0; i < perm.length; i++) {
        displacement += Math.abs(perm[i] - i);
    }
    return -displacement; // Maximize negative displacement
});

console.log(`Best fitness: ${permResult.bestFitness}`);
console.log(`Best permutation: [${permResult.bestGenome.join(', ')}]`);
permOptimizer.free();

// ============================================================================
// Multi-Objective Optimization (NSGA-II)
// ============================================================================

console.log('\n--- NSGA-II Multi-Objective Optimization ---');

const nsgaOptimizer = new wasm.Nsga2Optimizer(3, 2);
nsgaOptimizer.setPopulationSize(50);
nsgaOptimizer.setMaxGenerations(100);
nsgaOptimizer.setBounds(0, 1);
nsgaOptimizer.setSeed(42);

// Two conflicting objectives: minimize sum of squares vs minimize distance from 1
const nsgaResult = nsgaOptimizer.optimize((genome) => {
    const obj1 = genome.reduce((s, x) => s + x * x, 0);
    const obj2 = genome.reduce((s, x) => s + (1 - x) * (1 - x), 0);
    return [obj1, obj2];
});

console.log(`Pareto front size: ${nsgaResult.frontSize}`);
console.log(`Sample solutions:`);
for (let i = 0; i < Math.min(3, nsgaResult.frontSize); i++) {
    const sol = nsgaResult.getSolution(i);
    console.log(`  [${i + 1}] objectives: [${sol.objectives.map(o => o.toFixed(4)).join(', ')}]`);
}
nsgaOptimizer.free();

// ============================================================================
// Evolution Strategy
// ============================================================================

console.log('\n--- Evolution Strategy (15,100)-ES ---');

const esOptimizer = new wasm.EvolutionStrategyOptimizer(5);
esOptimizer.setMu(15);
esOptimizer.setLambda(100);
esOptimizer.setSelectionStrategy(wasm.ESSelection.MuCommaLambda);
esOptimizer.setInitialSigma(1.0);
esOptimizer.setSelfAdaptive(true);
esOptimizer.setMaxGenerations(200);
esOptimizer.setBounds(-5.12, 5.12);
esOptimizer.setSeed(42);

const esResult = esOptimizer.optimize((genome) => {
    // Rastrigin function (minimize)
    const A = 10;
    let sum = A * genome.length;
    for (const x of genome) {
        sum += x * x - A * Math.cos(2 * Math.PI * x);
    }
    return -sum; // Maximize negative
});

console.log(`Best fitness: ${(-esResult.bestFitness).toExponential(4)} (Rastrigin value)`);
console.log(`Solution: [${esResult.bestGenome.map(x => x.toFixed(4)).join(', ')}]`);
esOptimizer.free();

// ============================================================================
// Symbolic Regression
// ============================================================================

console.log('\n--- Symbolic Regression ---');

const srOptimizer = new wasm.SymbolicRegressionOptimizer();
srOptimizer.setPopulationSize(100);
srOptimizer.setMaxGenerations(50);
srOptimizer.setMaxTreeDepth(5);
srOptimizer.setSeed(42);

// Data for y = x^2
const xData = [];
const yData = [];
for (let x = -3; x <= 3; x += 0.5) {
    xData.push(x);
    yData.push(x * x);
}

const srResult = srOptimizer.optimizeData(new Float64Array(xData), new Float64Array(yData), 1);
console.log(`Best expression: ${srResult.expression}`);
console.log(`Best fitness (negative MSE): ${srResult.bestFitness.toExponential(4)}`);
console.log(`Tree depth: ${srResult.treeDepth}, size: ${srResult.treeSize}`);

console.log('\n' + '='.repeat(60));
console.log('All examples completed successfully!');
console.log('='.repeat(60));
