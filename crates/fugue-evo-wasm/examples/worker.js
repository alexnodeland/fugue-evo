/**
 * Web Worker for non-blocking optimization
 *
 * This worker runs optimization algorithms in a separate thread,
 * keeping the main UI responsive during long-running computations.
 */

// Import the WASM module
import init, {
    RealVectorOptimizer,
    BitStringOptimizer,
    UmdaOptimizer,
    Nsga2Optimizer,
    ZdtProblem,
    version
} from '../pkg/fugue_evo_wasm.js';

let wasmReady = false;

// Initialize WASM when worker starts
async function initWasm() {
    try {
        await init();
        wasmReady = true;
        self.postMessage({ type: 'ready', version: version() });
    } catch (e) {
        self.postMessage({ type: 'error', message: `Failed to initialize WASM: ${e.message}` });
    }
}

// Handle messages from main thread
self.onmessage = async function(e) {
    const { id, action, params } = e.data;

    if (!wasmReady) {
        self.postMessage({ id, type: 'error', message: 'WASM not ready' });
        return;
    }

    try {
        let result;
        const startTime = performance.now();

        switch (action) {
            case 'optimize-real-vector':
                result = runRealVectorOptimization(params);
                break;
            case 'optimize-bitstring':
                result = runBitStringOptimization(params);
                break;
            case 'optimize-umda':
                result = runUmdaOptimization(params);
                break;
            case 'optimize-nsga2':
                result = runNsga2Optimization(params);
                break;
            case 'optimize-zdt':
                result = runZdtOptimization(params);
                break;
            default:
                throw new Error(`Unknown action: ${action}`);
        }

        const elapsed = performance.now() - startTime;
        self.postMessage({
            id,
            type: 'result',
            result,
            elapsed
        });
    } catch (e) {
        self.postMessage({
            id,
            type: 'error',
            message: e.message || String(e)
        });
    }
};

function runRealVectorOptimization(params) {
    const {
        dimension = 10,
        populationSize = 100,
        maxGenerations = 100,
        fitness = 'sphere',
        lowerBound = -5.12,
        upperBound = 5.12,
        seed = 0
    } = params;

    const optimizer = new RealVectorOptimizer(dimension);
    optimizer.setPopulationSize(populationSize);
    optimizer.setMaxGenerations(maxGenerations);
    optimizer.setBounds(lowerBound, upperBound);
    optimizer.setFitness(fitness);
    if (seed) optimizer.setSeed(seed);

    const result = optimizer.optimize();
    const output = {
        bestFitness: result.bestFitness,
        bestGenome: Array.from(result.bestGenome),
        generations: result.generations,
        evaluations: result.evaluations
    };

    optimizer.free();
    return output;
}

function runBitStringOptimization(params) {
    const {
        length = 50,
        populationSize = 100,
        maxGenerations = 100,
        problem = 'onemax',
        schemaSize = 8,
        seed = 0
    } = params;

    const optimizer = new BitStringOptimizer(length);
    optimizer.setPopulationSize(populationSize);
    optimizer.setMaxGenerations(maxGenerations);
    if (seed) optimizer.setSeed(seed);

    let result;
    switch (problem) {
        case 'onemax':
            result = optimizer.solveOneMax();
            break;
        case 'leadingones':
            result = optimizer.solveLeadingOnes();
            break;
        case 'royalroad':
            result = optimizer.solveRoyalRoad(schemaSize);
            break;
        default:
            throw new Error(`Unknown problem: ${problem}`);
    }

    const output = {
        bestFitness: result.bestFitness,
        bestGenome: result.bestGenomeString(),
        generations: result.generations,
        evaluations: result.evaluations
    };

    optimizer.free();
    return output;
}

function runUmdaOptimization(params) {
    const {
        dimension = 10,
        populationSize = 100,
        maxGenerations = 100,
        fitness = 'sphere',
        selectionRatio = 0.5,
        lowerBound = -5.12,
        upperBound = 5.12,
        seed = 0
    } = params;

    const optimizer = new UmdaOptimizer(dimension);
    optimizer.setPopulationSize(populationSize);
    optimizer.setMaxGenerations(maxGenerations);
    optimizer.setSelectionRatio(selectionRatio);
    optimizer.setBounds(lowerBound, upperBound);
    if (seed) optimizer.setSeed(seed);

    const result = optimizer.optimize(fitness);
    const output = {
        bestFitness: result.bestFitness,
        bestGenome: Array.from(result.bestGenome),
        generations: result.generations,
        evaluations: result.evaluations
    };

    optimizer.free();
    return output;
}

function runNsga2Optimization(params) {
    const {
        dimension = 5,
        numObjectives = 2,
        populationSize = 50,
        maxGenerations = 100,
        lowerBound = -5,
        upperBound = 5,
        seed = 0
    } = params;

    const optimizer = new Nsga2Optimizer(dimension, numObjectives);
    optimizer.setPopulationSize(populationSize);
    optimizer.setMaxGenerations(maxGenerations);
    optimizer.setBounds(lowerBound, upperBound);
    if (seed) optimizer.setSeed(seed);

    // Use built-in ZDT1 as default multi-objective problem
    const result = optimizer.optimizeZdt(ZdtProblem.Zdt1);

    const paretoFront = [];
    for (let i = 0; i < result.frontSize; i++) {
        const sol = result.getSolution(i);
        paretoFront.push({
            genome: Array.from(sol.genome),
            objectives: Array.from(sol.objectives)
        });
    }

    const output = {
        frontSize: result.frontSize,
        paretoFront,
        generations: result.generations,
        evaluations: result.evaluations
    };

    optimizer.free();
    return output;
}

function runZdtOptimization(params) {
    const {
        problem = 'zdt1',
        dimension = 10,
        populationSize = 50,
        maxGenerations = 100,
        seed = 0
    } = params;

    const optimizer = new Nsga2Optimizer(dimension, 2);
    optimizer.setPopulationSize(populationSize);
    optimizer.setMaxGenerations(maxGenerations);
    if (seed) optimizer.setSeed(seed);

    let zdtProblem;
    switch (problem.toLowerCase()) {
        case 'zdt1':
            zdtProblem = ZdtProblem.Zdt1;
            break;
        case 'zdt2':
            zdtProblem = ZdtProblem.Zdt2;
            break;
        case 'zdt3':
            zdtProblem = ZdtProblem.Zdt3;
            break;
        default:
            throw new Error(`Unknown ZDT problem: ${problem}`);
    }

    const result = optimizer.optimizeZdt(zdtProblem);

    const paretoFront = [];
    for (let i = 0; i < result.frontSize; i++) {
        const sol = result.getSolution(i);
        paretoFront.push({
            genome: Array.from(sol.genome),
            objectives: Array.from(sol.objectives)
        });
    }

    const output = {
        problem,
        frontSize: result.frontSize,
        paretoFront,
        generations: result.generations,
        evaluations: result.evaluations
    };

    optimizer.free();
    return output;
}

// Start initialization
initWasm();
