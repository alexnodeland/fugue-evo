# WASM & Browser Usage

This guide shows how to use fugue-evo in web applications via WebAssembly.

## Overview

The `fugue-evo-wasm` package provides JavaScript/TypeScript bindings for running evolutionary optimization in the browser or Node.js.

## Installation

### NPM

```bash
npm install fugue-evo-wasm
```

### From Source

```bash
cd crates/fugue-evo-wasm
wasm-pack build --target web
```

## Basic Usage

### JavaScript

```javascript
import init, { SimpleGAOptimizer, OptimizationConfig } from 'fugue-evo-wasm';

async function runOptimization() {
  // Initialize WASM module
  await init();

  // Define configuration
  const config = new OptimizationConfig();
  config.population_size = 100;
  config.max_generations = 200;
  config.dimensions = 10;
  config.bounds_min = -5.12;
  config.bounds_max = 5.12;

  // Create optimizer
  const optimizer = new SimpleGAOptimizer(config);

  // Define fitness function
  const fitness = (genes) => {
    // Sphere function: minimize sum of squares
    return -genes.reduce((sum, x) => sum + x * x, 0);
  };

  // Run optimization
  const result = optimizer.run(fitness);

  console.log('Best fitness:', result.best_fitness);
  console.log('Best solution:', result.best_genome);
  console.log('Generations:', result.generations);
}

runOptimization();
```

### TypeScript

```typescript
import init, {
  SimpleGAOptimizer,
  OptimizationConfig,
  OptimizationResult
} from 'fugue-evo-wasm';

async function runOptimization(): Promise<void> {
  await init();

  const config: OptimizationConfig = {
    population_size: 100,
    max_generations: 200,
    dimensions: 10,
    bounds_min: -5.12,
    bounds_max: 5.12,
    mutation_rate: 0.1,
    crossover_rate: 0.9,
  };

  const optimizer = new SimpleGAOptimizer(config);

  const fitness = (genes: Float64Array): number => {
    let sum = 0;
    for (let i = 0; i < genes.length; i++) {
      sum += genes[i] * genes[i];
    }
    return -sum;
  };

  const result: OptimizationResult = optimizer.run(fitness);

  console.log(`Best fitness: ${result.best_fitness}`);
}
```

## Step-by-Step Evolution

For UI updates during evolution:

```javascript
import init, { SimpleGAOptimizer, OptimizationConfig } from 'fugue-evo-wasm';

async function interactiveEvolution() {
  await init();

  const config = new OptimizationConfig();
  config.population_size = 50;
  config.dimensions = 5;

  const optimizer = new SimpleGAOptimizer(config);

  const fitness = (genes) => -genes.reduce((s, x) => s + x * x, 0);

  // Initialize
  optimizer.initialize(fitness);

  // Step through evolution
  for (let gen = 0; gen < 100; gen++) {
    const state = optimizer.step(fitness);

    // Update UI
    document.getElementById('generation').textContent = gen;
    document.getElementById('best-fitness').textContent = state.best_fitness.toFixed(6);

    // Allow UI to update
    await new Promise(resolve => setTimeout(resolve, 10));
  }

  const result = optimizer.get_result();
  console.log('Final result:', result);
}
```

## Web Worker Integration

For heavy computations, use a Web Worker:

### main.js

```javascript
const worker = new Worker('optimizer-worker.js');

worker.onmessage = (event) => {
  const { type, data } = event.data;

  switch (type) {
    case 'progress':
      updateProgress(data.generation, data.best_fitness);
      break;
    case 'complete':
      displayResult(data.result);
      break;
  }
};

function startOptimization(config) {
  worker.postMessage({ type: 'start', config });
}
```

### optimizer-worker.js

```javascript
importScripts('fugue_evo_wasm.js');

let optimizer = null;

self.onmessage = async (event) => {
  const { type, config } = event.data;

  if (type === 'start') {
    await wasm_bindgen('fugue_evo_wasm_bg.wasm');

    const optConfig = new wasm_bindgen.OptimizationConfig();
    Object.assign(optConfig, config);

    optimizer = new wasm_bindgen.SimpleGAOptimizer(optConfig);

    const fitness = (genes) => {
      // Your fitness function
      return -genes.reduce((s, x) => s + x * x, 0);
    };

    optimizer.initialize(fitness);

    for (let gen = 0; gen < config.max_generations; gen++) {
      const state = optimizer.step(fitness);

      // Report progress every 10 generations
      if (gen % 10 === 0) {
        self.postMessage({
          type: 'progress',
          data: { generation: gen, best_fitness: state.best_fitness }
        });
      }
    }

    self.postMessage({
      type: 'complete',
      data: { result: optimizer.get_result() }
    });
  }
};
```

## Interactive Evolution in Browser

For human-in-the-loop optimization:

```javascript
import init, { InteractiveOptimizer, EvaluationMode } from 'fugue-evo-wasm';

class InteractiveEvolutionUI {
  constructor() {
    this.optimizer = null;
    this.currentCandidates = [];
  }

  async initialize() {
    await init();

    this.optimizer = new InteractiveOptimizer({
      population_size: 12,
      evaluation_mode: EvaluationMode.Rating,
      batch_size: 4,
    });

    this.optimizer.initialize();
    this.showNextBatch();
  }

  showNextBatch() {
    const request = this.optimizer.get_evaluation_request();

    if (request.type === 'complete') {
      this.showResults(request.result);
      return;
    }

    this.currentCandidates = request.candidates;
    this.renderCandidates(request.candidates);
  }

  renderCandidates(candidates) {
    const container = document.getElementById('candidates');
    container.innerHTML = '';

    candidates.forEach((candidate, index) => {
      const div = document.createElement('div');
      div.className = 'candidate';
      div.innerHTML = `
        <div class="visualization">${this.visualize(candidate.genome)}</div>
        <input type="range" min="1" max="10" value="5"
               data-id="${candidate.id}" class="rating">
      `;
      container.appendChild(div);
    });
  }

  submitRatings() {
    const ratings = [];
    document.querySelectorAll('.rating').forEach(input => {
      ratings.push({
        id: parseInt(input.dataset.id),
        rating: parseFloat(input.value)
      });
    });

    this.optimizer.provide_ratings(ratings);
    this.showNextBatch();
  }

  visualize(genome) {
    // Convert genome to visual representation
    const colors = genome.map(g => {
      const hue = ((g + 5) / 10) * 360;
      return `hsl(${hue}, 70%, 50%)`;
    });
    return colors.map(c => `<span style="background:${c}">■</span>`).join('');
  }

  showResults(result) {
    console.log('Evolution complete!', result);
  }
}
```

## Performance Considerations

### Memory Management

WASM has limited memory. For large populations:

```javascript
// Free memory when done
optimizer.free();
config.free();
```

### Typed Arrays

Use typed arrays for efficiency:

```javascript
// Efficient: Float64Array
const genes = new Float64Array([1.0, 2.0, 3.0]);

// Less efficient: regular array (converted internally)
const genes = [1.0, 2.0, 3.0];
```

### Batch Fitness Evaluation

Reduce JS↔WASM calls:

```javascript
// Less efficient: evaluate one at a time
for (const individual of population) {
  const fitness = evaluateFitness(individual);
}

// More efficient: batch evaluation
const fitnesses = evaluateBatch(population);
```

## Framework Integration

### React

```jsx
import { useEffect, useState } from 'react';
import init, { SimpleGAOptimizer, OptimizationConfig } from 'fugue-evo-wasm';

function EvolutionComponent() {
  const [result, setResult] = useState(null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let cancelled = false;

    async function runEvolution() {
      await init();

      const config = new OptimizationConfig();
      config.population_size = 100;
      config.dimensions = 10;
      config.max_generations = 100;

      const optimizer = new SimpleGAOptimizer(config);
      const fitness = (genes) => -genes.reduce((s, x) => s + x * x, 0);

      optimizer.initialize(fitness);

      for (let gen = 0; gen < 100 && !cancelled; gen++) {
        optimizer.step(fitness);
        setProgress(gen + 1);
        await new Promise(r => setTimeout(r, 0)); // Yield to React
      }

      if (!cancelled) {
        setResult(optimizer.get_result());
      }

      optimizer.free();
      config.free();
    }

    runEvolution();
    return () => { cancelled = true; };
  }, []);

  return (
    <div>
      <p>Progress: {progress}/100</p>
      {result && <p>Best fitness: {result.best_fitness}</p>}
    </div>
  );
}
```

### Vue

```vue
<template>
  <div>
    <p>Progress: {{ progress }}/{{ maxGenerations }}</p>
    <p v-if="result">Best fitness: {{ result.best_fitness }}</p>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import init, { SimpleGAOptimizer, OptimizationConfig } from 'fugue-evo-wasm';

const progress = ref(0);
const maxGenerations = ref(100);
const result = ref(null);
let cancelled = false;

onMounted(async () => {
  await init();

  const config = new OptimizationConfig();
  config.population_size = 100;
  config.dimensions = 10;
  config.max_generations = maxGenerations.value;

  const optimizer = new SimpleGAOptimizer(config);
  const fitness = (genes) => -genes.reduce((s, x) => s + x * x, 0);

  optimizer.initialize(fitness);

  for (let gen = 0; gen < maxGenerations.value && !cancelled; gen++) {
    optimizer.step(fitness);
    progress.value = gen + 1;
    await new Promise(r => setTimeout(r, 0));
  }

  if (!cancelled) {
    result.value = optimizer.get_result();
  }

  optimizer.free();
  config.free();
});

onUnmounted(() => {
  cancelled = true;
});
</script>
```

## Limitations

- **No file I/O**: Can't use checkpointing
- **Single-threaded**: No Rayon parallelism
- **Memory limits**: ~4GB maximum
- **Fitness callback overhead**: JS calls add latency

## Next Steps

- [Interactive Evolution](../tutorials/interactive-evolution.md) - Human-in-the-loop in browser
- [Custom Fitness Functions](./custom-fitness.md) - Optimize your problem
