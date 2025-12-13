# fugue-evo-wasm

WebAssembly bindings for the fugue-evo genetic algorithm library.

## Features

- **Real Vector Optimization** - Optimize continuous functions with genetic algorithms
- **CMA-ES** - Covariance Matrix Adaptation Evolution Strategy
- **NSGA-II** - Multi-objective optimization with Pareto fronts
- **Interactive Evolution** - Human-in-the-loop optimization
- **Symbolic Regression** - Evolve mathematical expressions
- **Custom Fitness Functions** - Define fitness in JavaScript

## Installation

### From npm

```bash
npm install fugue-evo-wasm
```

### Build from source

```bash
# Install wasm-pack
cargo install wasm-pack

# Build the package
wasm-pack build --target web

# For npm publishing
wasm-pack build --target bundler
```

## Usage

### Browser (ES Modules)

```html
<script type="module">
import init, { RealVectorOptimizer } from './pkg/fugue_evo_wasm.js';

await init();

const optimizer = new RealVectorOptimizer(10);
optimizer.setPopulationSize(100);
optimizer.setBounds(-5.12, 5.12);
optimizer.setFitness("rastrigin");

const result = optimizer.optimize(200);
console.log("Best fitness:", result.best_fitness);
console.log("Best solution:", result.best_genome);

optimizer.free();
</script>
```

### Node.js

```javascript
const { RealVectorOptimizer } = require('fugue-evo-wasm');

const optimizer = new RealVectorOptimizer(10);
optimizer.setPopulationSize(100);
optimizer.setBounds(-5.12, 5.12);
optimizer.setFitness("sphere");

const result = optimizer.optimize(100);
console.log(result);
```

### Custom Fitness Functions

```javascript
const optimizer = new RealVectorOptimizer(5);
optimizer.setPopulationSize(50);
optimizer.setBounds(-10, 10);

// Define custom fitness in JavaScript
optimizer.setCustomFitness((genome) => {
    return genome.reduce((sum, x) => sum + x * x, 0);
});

const result = optimizer.optimize(100);
```

### CMA-ES Optimization

```javascript
import { CmaEsOptimizer } from 'fugue-evo-wasm';

const optimizer = new CmaEsOptimizer(5);
optimizer.setSigma(0.5);
optimizer.setCustomFitness((x) => {
    // Rosenbrock function
    let sum = 0;
    for (let i = 0; i < x.length - 1; i++) {
        sum += 100 * Math.pow(x[i+1] - x[i]*x[i], 2) + Math.pow(1 - x[i], 2);
    }
    return sum;
});

const result = optimizer.optimize(500);
```

### Multi-Objective Optimization (NSGA-II)

```javascript
import { Nsga2Optimizer } from 'fugue-evo-wasm';

const optimizer = new Nsga2Optimizer(3);
optimizer.setPopulationSize(50);
optimizer.setBounds(0, 1);

optimizer.setObjectives([
    (x) => x.reduce((s, xi) => s + xi * xi, 0),  // Minimize sum of squares
    (x) => x.reduce((s, xi) => s + (1-xi) * (1-xi), 0)  // Minimize distance from 1
]);

const result = optimizer.optimize(100);
console.log("Pareto front size:", result.pareto_front.length);
```

### Interactive Evolution

```javascript
import { InteractiveOptimizer } from 'fugue-evo-wasm';

// Mode: 0=Rating, 1=Pairwise, 2=BatchSelection
const optimizer = InteractiveOptimizer.withConfig(5, 10, 0);
optimizer.setBounds(-5, 5);

while (true) {
    const step = optimizer.step();

    if (step.result_type === 'complete') {
        console.log("Best:", step.best_genome);
        break;
    }

    if (step.result_type === 'needs_evaluation') {
        const request = step.request;

        if (request.request_type === 'rating') {
            // Rate candidates 1-5
            const ratings = getUserRatings(request.candidates);
            optimizer.provideRatings(ratings);
        } else if (request.request_type === 'pairwise') {
            // Choose between two candidates
            const winnerId = getUserChoice(request.candidates);
            optimizer.providePairwiseChoice(winnerId);
        } else {
            // Select favorites from batch
            const selectedIds = getUserSelections(request.candidates);
            optimizer.provideBatchSelection(selectedIds);
        }
    }
}
```

## API Reference

### RealVectorOptimizer

| Method | Description |
|--------|-------------|
| `new(dimensions)` | Create optimizer with given dimensions |
| `setPopulationSize(n)` | Set population size |
| `setBounds(min, max)` | Set search bounds |
| `setFitness(name)` | Use built-in fitness: "sphere", "rastrigin", "rosenbrock", "ackley" |
| `setCustomFitness(fn)` | Set JavaScript fitness function |
| `optimize(generations)` | Run optimization |

### CmaEsOptimizer

| Method | Description |
|--------|-------------|
| `new(dimensions)` | Create CMA-ES optimizer |
| `setSigma(sigma)` | Set initial step size |
| `setCustomFitness(fn)` | Set fitness function |
| `optimize(generations)` | Run optimization |

### Nsga2Optimizer

| Method | Description |
|--------|-------------|
| `new(dimensions)` | Create NSGA-II optimizer |
| `setPopulationSize(n)` | Set population size |
| `setBounds(min, max)` | Set search bounds |
| `setObjectives(fns)` | Set array of objective functions |
| `optimize(generations)` | Run optimization |

### InteractiveOptimizer

| Method | Description |
|--------|-------------|
| `new(dimensions)` | Create with defaults |
| `withConfig(dims, pop, mode)` | Create with configuration |
| `setBounds(min, max)` | Set search bounds |
| `step()` | Advance one step |
| `provideRatings(ratings)` | Submit ratings (1-5 scale) |
| `providePairwiseChoice(id)` | Submit pairwise choice |
| `provideBatchSelection(ids)` | Submit batch selections |

## Running the Example

```bash
# Build the WASM package
cd crates/fugue-evo-wasm
wasm-pack build --target web

# Serve the examples directory
python -m http.server 8080 --directory examples

# Open http://localhost:8080 in your browser
```

## License

MIT OR Apache-2.0
