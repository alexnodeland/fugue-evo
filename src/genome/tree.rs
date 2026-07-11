//! Tree genomes for genetic programming
//!
//! This module provides tree-based genomes for symbolic regression and
//! genetic programming applications.
//!
//! # Deep trees and stack safety
//!
//! The read-side traversals ([`TreeNode::depth`], [`TreeNode::size`] and
//! [`TreeGenome::evaluate`]) use explicit work stacks instead of recursion, so
//! they do not overflow the call stack even for pathologically deep trees.
//!
//! Tearing a tree down, however, cannot be made stack-safe via a `Drop` impl:
//! adding `Drop` to [`TreeNode`] (or [`TreeGenome`]) is rejected by the compiler
//! (error E0509) at the points where the operator layer
//! (`crate::operators`, a separate work package) moves values out of an owned
//! `TreeNode`/`TreeGenome`. Rust's compiler-generated drop glue is therefore
//! still recursive. Callers that construct or deserialize pathologically deep
//! trees should hand them to [`TreeGenome::dismantle`] (or
//! [`drop_node_iteratively`]) to free them without recursion instead of letting
//! them drop implicitly.

use fugue::{addr, ChoiceValue, Trace};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::GenomeError;
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::EvolutionaryGenome;

/// A node in a GP tree
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum TreeNode<T: Terminal, F: Function> {
    /// Terminal node (leaf)
    Terminal(T),
    /// Function node (internal)
    Function(F, Vec<TreeNode<T, F>>),
}

impl<T: Terminal, F: Function> TreeNode<T, F> {
    /// Create a new terminal node
    pub fn terminal(value: T) -> Self {
        Self::Terminal(value)
    }

    /// Create a new function node
    pub fn function(func: F, children: Vec<Self>) -> Self {
        Self::Function(func, children)
    }

    /// Check if this is a terminal node
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal(_))
    }

    /// Check if this is a function node
    pub fn is_function(&self) -> bool {
        matches!(self, Self::Function(_, _))
    }

    /// Get the depth of this subtree.
    ///
    /// Uses an explicit work stack rather than recursion so that pathologically
    /// deep trees cannot overflow the call stack (see the module note on deep
    /// trees).
    pub fn depth(&self) -> usize {
        let mut max_depth = 0;
        // (node, depth-of-node) pairs; a terminal has depth 1.
        let mut stack: Vec<(&Self, usize)> = vec![(self, 1)];
        while let Some((node, d)) = stack.pop() {
            if d > max_depth {
                max_depth = d;
            }
            if let Self::Function(_, children) = node {
                for child in children {
                    stack.push((child, d + 1));
                }
            }
        }
        max_depth
    }

    /// Get the number of nodes in this subtree.
    ///
    /// Uses an explicit work stack rather than recursion (see the module note on
    /// deep trees).
    pub fn size(&self) -> usize {
        let mut count = 0;
        let mut stack: Vec<&Self> = vec![self];
        while let Some(node) = stack.pop() {
            count += 1;
            if let Self::Function(_, children) = node {
                for child in children {
                    stack.push(child);
                }
            }
        }
        count
    }

    /// Get all node positions (preorder traversal indices)
    pub fn positions(&self) -> Vec<Vec<usize>> {
        let mut positions = Vec::new();
        self.collect_positions(&[], &mut positions);
        positions
    }

    fn collect_positions(&self, path: &[usize], positions: &mut Vec<Vec<usize>>) {
        positions.push(path.to_vec());
        if let Self::Function(_, children) = self {
            for (i, child) in children.iter().enumerate() {
                let mut child_path = path.to_vec();
                child_path.push(i);
                child.collect_positions(&child_path, positions);
            }
        }
    }

    /// Get a subtree at the given path
    pub fn get_subtree(&self, path: &[usize]) -> Option<&Self> {
        if path.is_empty() {
            return Some(self);
        }

        if let Self::Function(_, children) = self {
            let idx = path[0];
            if idx < children.len() {
                children[idx].get_subtree(&path[1..])
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get a mutable subtree at the given path
    pub fn get_subtree_mut(&mut self, path: &[usize]) -> Option<&mut Self> {
        if path.is_empty() {
            return Some(self);
        }

        if let Self::Function(_, children) = self {
            let idx = path[0];
            if idx < children.len() {
                children[idx].get_subtree_mut(&path[1..])
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Replace a subtree at the given path
    pub fn replace_subtree(&mut self, path: &[usize], new_subtree: Self) -> bool {
        if path.is_empty() {
            *self = new_subtree;
            return true;
        }

        if let Self::Function(_, children) = self {
            let idx = path[0];
            if idx < children.len() {
                if path.len() == 1 {
                    children[idx] = new_subtree;
                    true
                } else {
                    children[idx].replace_subtree(&path[1..], new_subtree)
                }
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Get all terminal positions
    pub fn terminal_positions(&self) -> Vec<Vec<usize>> {
        let mut positions = Vec::new();
        self.collect_terminal_positions(&[], &mut positions);
        positions
    }

    fn collect_terminal_positions(&self, path: &[usize], positions: &mut Vec<Vec<usize>>) {
        match self {
            Self::Terminal(_) => positions.push(path.to_vec()),
            Self::Function(_, children) => {
                for (i, child) in children.iter().enumerate() {
                    let mut child_path = path.to_vec();
                    child_path.push(i);
                    child.collect_terminal_positions(&child_path, positions);
                }
            }
        }
    }

    /// Get all function positions
    pub fn function_positions(&self) -> Vec<Vec<usize>> {
        let mut positions = Vec::new();
        self.collect_function_positions(&[], &mut positions);
        positions
    }

    fn collect_function_positions(&self, path: &[usize], positions: &mut Vec<Vec<usize>>) {
        if let Self::Function(_, children) = self {
            positions.push(path.to_vec());
            for (i, child) in children.iter().enumerate() {
                let mut child_path = path.to_vec();
                child_path.push(i);
                child.collect_function_positions(&child_path, positions);
            }
        }
    }
}

/// Trait for terminal nodes in GP trees
pub trait Terminal:
    Clone + Send + Sync + PartialEq + fmt::Debug + Serialize + for<'de> Deserialize<'de> + 'static
{
    /// Generate a random terminal
    fn random<R: Rng>(rng: &mut R) -> Self;

    /// Get the set of available terminals
    fn terminals() -> &'static [Self];

    /// Evaluate this terminal with the given variable bindings
    fn evaluate(&self, variables: &[f64]) -> f64;

    /// Convert to string representation
    fn to_string(&self) -> String;

    /// Encode this terminal as a `(type_code, payload)` pair for lossless trace
    /// round-tripping.
    ///
    /// `type_code` identifies the terminal variant (a discriminant) and
    /// `payload` carries its associated value. The pair must satisfy the
    /// inverse relationship `Self::decode(self.encode()) == *self` so that
    /// [`TreeGenome::from_trace`](crate::genome::traits::EvolutionaryGenome::from_trace)
    /// reproduces the exact terminal that
    /// [`TreeGenome::to_trace`](crate::genome::traits::EvolutionaryGenome::to_trace)
    /// serialized.
    fn encode(&self) -> (f64, f64);

    /// Decode a terminal previously produced by [`encode`](Self::encode).
    ///
    /// This is the inverse of [`encode`](Self::encode) and must reconstruct an
    /// equal terminal for any `(type_code, payload)` this type emits.
    fn decode(type_code: f64, payload: f64) -> Self;
}

/// Trait for function nodes in GP trees
pub trait Function:
    Clone + Send + Sync + PartialEq + fmt::Debug + Serialize + for<'de> Deserialize<'de> + 'static
{
    /// Get the arity (number of arguments) of this function
    fn arity(&self) -> usize;

    /// Generate a random function
    fn random<R: Rng>(rng: &mut R) -> Self;

    /// Get the set of available functions
    fn functions() -> &'static [Self];

    /// Apply this function to the given arguments
    fn apply(&self, args: &[f64]) -> f64;

    /// Convert to string representation
    fn to_string(&self) -> String;
}

/// Standard arithmetic terminals for symbolic regression
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ArithmeticTerminal {
    /// Variable x_i
    Variable(usize),
    /// Constant value
    Constant(f64),
    /// Ephemeral random constant (ERC)
    Erc(f64),
}

impl Terminal for ArithmeticTerminal {
    fn random<R: Rng>(rng: &mut R) -> Self {
        let choice: u8 = rng.gen_range(0..3);
        match choice {
            0 => Self::Variable(rng.gen_range(0..10)),
            1 => Self::Constant(rng.gen_range(-10.0..10.0)),
            _ => Self::Erc(rng.gen_range(-1.0..1.0)),
        }
    }

    fn terminals() -> &'static [Self] {
        // Return a representative set; actual terminals depend on context
        &[]
    }

    fn evaluate(&self, variables: &[f64]) -> f64 {
        match self {
            Self::Variable(i) => variables.get(*i).copied().unwrap_or(0.0),
            Self::Constant(c) | Self::Erc(c) => *c,
        }
    }

    fn to_string(&self) -> String {
        match self {
            Self::Variable(i) => format!("x{}", i),
            Self::Constant(c) | Self::Erc(c) => format!("{:.4}", c),
        }
    }

    fn encode(&self) -> (f64, f64) {
        // type_code: 0 = Variable, 1 = Constant, 2 = Erc
        match self {
            Self::Variable(i) => (0.0, *i as f64),
            Self::Constant(c) => (1.0, *c),
            Self::Erc(c) => (2.0, *c),
        }
    }

    fn decode(type_code: f64, payload: f64) -> Self {
        match type_code.round() as i64 {
            0 => Self::Variable(payload.max(0.0) as usize),
            1 => Self::Constant(payload),
            2 => Self::Erc(payload),
            // Unknown discriminant (corrupt trace): preserve the payload as a
            // constant rather than fabricating a random terminal.
            _ => Self::Constant(payload),
        }
    }
}

/// Standard arithmetic functions for symbolic regression
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ArithmeticFunction {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Protected division (returns 1.0 for division by zero)
    Div,
    /// Sine
    Sin,
    /// Cosine
    Cos,
    /// Exponential
    Exp,
    /// Natural logarithm (protected)
    Log,
    /// Square root (protected)
    Sqrt,
    /// Power
    Pow,
    /// Negation (unary)
    Neg,
    /// Absolute value (unary)
    Abs,
}

impl Function for ArithmeticFunction {
    fn arity(&self) -> usize {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Pow => 2,
            Self::Sin | Self::Cos | Self::Exp | Self::Log | Self::Sqrt | Self::Neg | Self::Abs => 1,
        }
    }

    fn random<R: Rng>(rng: &mut R) -> Self {
        let funcs = Self::functions();
        funcs[rng.gen_range(0..funcs.len())].clone()
    }

    fn functions() -> &'static [Self] {
        &[
            Self::Add,
            Self::Sub,
            Self::Mul,
            Self::Div,
            Self::Sin,
            Self::Cos,
            Self::Exp,
            Self::Log,
            Self::Sqrt,
            Self::Neg,
            Self::Abs,
        ]
    }

    fn apply(&self, args: &[f64]) -> f64 {
        match self {
            Self::Add => args.get(0).unwrap_or(&0.0) + args.get(1).unwrap_or(&0.0),
            Self::Sub => args.get(0).unwrap_or(&0.0) - args.get(1).unwrap_or(&0.0),
            Self::Mul => args.get(0).unwrap_or(&1.0) * args.get(1).unwrap_or(&1.0),
            Self::Div => {
                let a = args.get(0).unwrap_or(&0.0);
                let b = args.get(1).unwrap_or(&1.0);
                if b.abs() < 1e-10 {
                    1.0 // Protected division
                } else {
                    a / b
                }
            }
            Self::Sin => args.get(0).unwrap_or(&0.0).sin(),
            Self::Cos => args.get(0).unwrap_or(&0.0).cos(),
            Self::Exp => {
                let x = args.get(0).unwrap_or(&0.0);
                if *x > 700.0 {
                    f64::MAX // Overflow protection
                } else {
                    x.exp()
                }
            }
            Self::Log => {
                let x = args.get(0).unwrap_or(&1.0);
                if *x <= 0.0 {
                    0.0 // Protected log
                } else {
                    x.ln()
                }
            }
            Self::Sqrt => {
                let x = args.get(0).unwrap_or(&0.0);
                if *x < 0.0 {
                    (-x).sqrt() // Protected sqrt
                } else {
                    x.sqrt()
                }
            }
            Self::Pow => {
                let base = args.get(0).unwrap_or(&1.0);
                let exp = args.get(1).unwrap_or(&1.0);
                // Protected power
                if base.abs() < 1e-10 && *exp < 0.0 {
                    0.0
                } else {
                    base.powf(*exp).clamp(-1e10, 1e10)
                }
            }
            Self::Neg => -args.get(0).unwrap_or(&0.0),
            Self::Abs => args.get(0).unwrap_or(&0.0).abs(),
        }
    }

    fn to_string(&self) -> String {
        match self {
            Self::Add => "+".to_string(),
            Self::Sub => "-".to_string(),
            Self::Mul => "*".to_string(),
            Self::Div => "/".to_string(),
            Self::Sin => "sin".to_string(),
            Self::Cos => "cos".to_string(),
            Self::Exp => "exp".to_string(),
            Self::Log => "log".to_string(),
            Self::Sqrt => "sqrt".to_string(),
            Self::Pow => "pow".to_string(),
            Self::Neg => "neg".to_string(),
            Self::Abs => "abs".to_string(),
        }
    }
}

/// Tree genome for genetic programming
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TreeGenome<T: Terminal = ArithmeticTerminal, F: Function = ArithmeticFunction> {
    /// Root node of the tree
    pub root: TreeNode<T, F>,
    /// Maximum allowed depth
    pub max_depth: usize,
}

impl<T: Terminal, F: Function> TreeGenome<T, F> {
    /// Create a new tree genome
    pub fn new(root: TreeNode<T, F>, max_depth: usize) -> Self {
        Self { root, max_depth }
    }

    /// Get the depth of the tree
    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    /// Get the number of nodes in the tree
    pub fn size(&self) -> usize {
        self.root.size()
    }

    /// Evaluate the tree with given variable bindings.
    ///
    /// Uses an explicit work stack (iterative post-order traversal) rather than
    /// recursion, so a pathologically deep tree cannot overflow the call stack
    /// (see the module note on deep trees).
    pub fn evaluate(&self, variables: &[f64]) -> f64 {
        // Two task kinds: `Eval` expands a node; `Apply` combines the results of
        // a function node's already-evaluated children.
        enum Task<'a, T: Terminal, F: Function> {
            Eval(&'a TreeNode<T, F>),
            Apply(&'a F, usize),
        }

        let mut tasks: Vec<Task<T, F>> = vec![Task::Eval(&self.root)];
        let mut values: Vec<f64> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                Task::Eval(node) => match node {
                    TreeNode::Terminal(t) => values.push(t.evaluate(variables)),
                    TreeNode::Function(f, children) => {
                        // Schedule the apply, then push children in reverse so
                        // they evaluate left-to-right and land on `values` in
                        // argument order.
                        tasks.push(Task::Apply(f, children.len()));
                        for child in children.iter().rev() {
                            tasks.push(Task::Eval(child));
                        }
                    }
                },
                Task::Apply(f, arity) => {
                    let start = values.len() - arity;
                    let args = values.split_off(start);
                    values.push(f.apply(&args));
                }
            }
        }

        values.pop().unwrap_or(0.0)
    }

    /// Free this tree without deep recursion.
    ///
    /// Consumes the genome and dismantles its tree iteratively (see the module
    /// note on deep trees). Use this for pathologically deep trees that would
    /// otherwise overflow the stack when dropped implicitly.
    pub fn dismantle(self) {
        drop_node_iteratively(self.root);
    }

    /// Generate a random tree using the "full" method
    pub fn generate_full<R: Rng>(rng: &mut R, depth: usize, max_depth: usize) -> Self {
        let root = Self::generate_full_node(rng, depth, 0);
        Self { root, max_depth }
    }

    fn generate_full_node<R: Rng>(
        rng: &mut R,
        target_depth: usize,
        current_depth: usize,
    ) -> TreeNode<T, F> {
        if current_depth >= target_depth {
            TreeNode::Terminal(T::random(rng))
        } else {
            let func = F::random(rng);
            let arity = func.arity();
            let children: Vec<TreeNode<T, F>> = (0..arity)
                .map(|_| Self::generate_full_node(rng, target_depth, current_depth + 1))
                .collect();
            TreeNode::Function(func, children)
        }
    }

    /// Generate a random tree using the "grow" method
    pub fn generate_grow<R: Rng>(rng: &mut R, max_depth: usize, terminal_prob: f64) -> Self {
        let root = Self::generate_grow_node(rng, max_depth, 0, terminal_prob);
        Self { root, max_depth }
    }

    fn generate_grow_node<R: Rng>(
        rng: &mut R,
        max_depth: usize,
        current_depth: usize,
        terminal_prob: f64,
    ) -> TreeNode<T, F> {
        if current_depth >= max_depth {
            TreeNode::Terminal(T::random(rng))
        } else if rng.gen::<f64>() < terminal_prob {
            TreeNode::Terminal(T::random(rng))
        } else {
            let func = F::random(rng);
            let arity = func.arity();
            let children: Vec<TreeNode<T, F>> = (0..arity)
                .map(|_| Self::generate_grow_node(rng, max_depth, current_depth + 1, terminal_prob))
                .collect();
            TreeNode::Function(func, children)
        }
    }

    /// Generate using ramped half-and-half
    pub fn generate_ramped_half_and_half<R: Rng>(
        rng: &mut R,
        min_depth: usize,
        max_depth: usize,
    ) -> Self {
        let depth = rng.gen_range(min_depth..=max_depth);
        if rng.gen() {
            Self::generate_full(rng, depth, max_depth)
        } else {
            Self::generate_grow(rng, depth, 0.3)
        }
    }

    /// Generate a random tree with an explicit maximum depth.
    ///
    /// This is the honest constructor for random generation: unlike
    /// [`EvolutionaryGenome::generate`](crate::genome::traits::EvolutionaryGenome::generate),
    /// which overloads `MultiBounds` and remaps its *dimension count* to a depth,
    /// this takes the maximum depth directly. It uses ramped half-and-half
    /// between depth 2 and `max_depth` (both clamped to at least 1).
    pub fn generate_with_depth<R: Rng>(rng: &mut R, max_depth: usize) -> Self {
        let max_depth = max_depth.max(1);
        let min_depth = 2.min(max_depth);
        Self::generate_ramped_half_and_half(rng, min_depth, max_depth)
    }

    /// Convert tree to S-expression string
    pub fn to_sexpr(&self) -> String {
        self.node_to_sexpr(&self.root)
    }

    fn node_to_sexpr(&self, node: &TreeNode<T, F>) -> String {
        match node {
            TreeNode::Terminal(t) => t.to_string(),
            TreeNode::Function(f, children) => {
                let child_strs: Vec<String> =
                    children.iter().map(|c| self.node_to_sexpr(c)).collect();
                format!("({} {})", f.to_string(), child_strs.join(" "))
            }
        }
    }

    /// Get a random node position
    pub fn random_position<R: Rng>(&self, rng: &mut R) -> Vec<usize> {
        let positions = self.root.positions();
        positions[rng.gen_range(0..positions.len())].clone()
    }

    /// Get a random terminal position
    pub fn random_terminal_position<R: Rng>(&self, rng: &mut R) -> Option<Vec<usize>> {
        let positions = self.root.terminal_positions();
        if positions.is_empty() {
            None
        } else {
            Some(positions[rng.gen_range(0..positions.len())].clone())
        }
    }

    /// Get a random function position
    pub fn random_function_position<R: Rng>(&self, rng: &mut R) -> Option<Vec<usize>> {
        let positions = self.root.function_positions();
        if positions.is_empty() {
            None
        } else {
            Some(positions[rng.gen_range(0..positions.len())].clone())
        }
    }
}

impl<T: Terminal, F: Function> EvolutionaryGenome for TreeGenome<T, F> {
    type Allele = TreeNode<T, F>;
    type Phenotype = Self;

    fn to_trace(&self) -> Trace {
        let mut trace = Trace::default();
        let mut index = 0;
        self.node_to_trace(&self.root, &mut trace, &mut index);
        // Store max_depth and total size
        trace.insert_choice(
            addr!("tree_max_depth"),
            ChoiceValue::Usize(self.max_depth),
            0.0,
        );
        trace.insert_choice(addr!("tree_size"), ChoiceValue::Usize(index), 0.0);
        trace
    }

    fn from_trace(trace: &Trace) -> Result<Self, GenomeError> {
        let max_depth = trace
            .get_usize(&addr!("tree_max_depth"))
            .ok_or_else(|| GenomeError::MissingAddress("tree_max_depth".to_string()))?;

        let mut index = 0;
        let root = Self::node_from_trace(trace, &mut index)?;
        Ok(Self { root, max_depth })
    }

    fn decode(&self) -> Self::Phenotype {
        self.clone()
    }

    fn dimension(&self) -> usize {
        self.size()
    }

    /// Generate a random tree.
    ///
    /// Only `bounds.dimension()` is consulted — it is remapped (clamped to
    /// `[3, 10]`) to a maximum tree depth — and the per-dimension `min`/`max`
    /// values are ignored. Prefer [`TreeGenome::generate_with_depth`] to make the
    /// depth explicit instead of overloading `MultiBounds`.
    fn generate<R: Rng>(rng: &mut R, bounds: &MultiBounds) -> Self {
        let max_depth = bounds.dimension().clamp(3, 10);
        Self::generate_with_depth(rng, max_depth)
    }

    fn distance(&self, other: &Self) -> f64 {
        // Tree edit distance approximation based on size difference
        let size_diff = (self.size() as f64 - other.size() as f64).abs();
        let depth_diff = (self.depth() as f64 - other.depth() as f64).abs();
        size_diff + depth_diff
    }

    fn try_distance(&self, other: &Self) -> Result<f64, GenomeError> {
        // Any two trees are comparable (size/depth deltas), so this never errs.
        Ok(self.distance(other))
    }

    fn trace_prefix() -> &'static str {
        "tree"
    }
}

impl<T: Terminal, F: Function> TreeGenome<T, F> {
    fn node_to_trace(&self, node: &TreeNode<T, F>, trace: &mut Trace, index: &mut usize) {
        let current_index = *index;
        *index += 1;

        match node {
            TreeNode::Terminal(t) => {
                // Store is_terminal flag (true = terminal)
                trace.insert_choice(
                    addr!("tree_is_terminal", current_index),
                    ChoiceValue::Bool(true),
                    0.0,
                );
                // For ArithmeticTerminal, store the variant type and value
                // We encode using f64 for simplicity
                let (term_type, term_val) = Self::encode_terminal(t);
                trace.insert_choice(
                    addr!("tree_term_type", current_index),
                    ChoiceValue::F64(term_type),
                    0.0,
                );
                trace.insert_choice(
                    addr!("tree_term_val", current_index),
                    ChoiceValue::F64(term_val),
                    0.0,
                );
            }
            TreeNode::Function(f, children) => {
                // Store is_terminal flag (false = function)
                trace.insert_choice(
                    addr!("tree_is_terminal", current_index),
                    ChoiceValue::Bool(false),
                    0.0,
                );
                // Store function type as index and arity
                let func_idx = Self::encode_function(f);
                trace.insert_choice(
                    addr!("tree_func_idx", current_index),
                    ChoiceValue::Usize(func_idx),
                    0.0,
                );
                trace.insert_choice(
                    addr!("tree_arity", current_index),
                    ChoiceValue::Usize(children.len()),
                    0.0,
                );
                // Recurse into children
                for child in children {
                    self.node_to_trace(child, trace, index);
                }
            }
        }
    }

    fn node_from_trace(trace: &Trace, index: &mut usize) -> Result<TreeNode<T, F>, GenomeError> {
        let current_index = *index;
        *index += 1;

        let is_terminal = trace
            .get_bool(&addr!("tree_is_terminal", current_index))
            .ok_or_else(|| {
                GenomeError::MissingAddress(format!("tree_is_terminal#{}", current_index))
            })?;

        if is_terminal {
            let term_type = trace
                .get_f64(&addr!("tree_term_type", current_index))
                .ok_or_else(|| {
                    GenomeError::MissingAddress(format!("tree_term_type#{}", current_index))
                })?;
            let term_val = trace
                .get_f64(&addr!("tree_term_val", current_index))
                .ok_or_else(|| {
                    GenomeError::MissingAddress(format!("tree_term_val#{}", current_index))
                })?;

            let terminal = Self::decode_terminal(term_type, term_val)?;
            Ok(TreeNode::Terminal(terminal))
        } else {
            let func_idx = trace
                .get_usize(&addr!("tree_func_idx", current_index))
                .ok_or_else(|| {
                    GenomeError::MissingAddress(format!("tree_func_idx#{}", current_index))
                })?;
            let arity = trace
                .get_usize(&addr!("tree_arity", current_index))
                .ok_or_else(|| {
                    GenomeError::MissingAddress(format!("tree_arity#{}", current_index))
                })?;

            let func = Self::decode_function(func_idx)?;
            let mut children = Vec::with_capacity(arity);
            for _ in 0..arity {
                children.push(Self::node_from_trace(trace, index)?);
            }
            Ok(TreeNode::Function(func, children))
        }
    }

    // Encode a terminal as a (type_code, payload) pair via the type's own
    // lossless `Terminal::encode`, so trace round-tripping preserves the exact
    // terminal (variant + value) rather than fabricating a random one.
    fn encode_terminal(terminal: &T) -> (f64, f64) {
        terminal.encode()
    }

    fn decode_terminal(term_type: f64, term_val: f64) -> Result<T, GenomeError> {
        Ok(T::decode(term_type, term_val))
    }

    // Encode a function as its index in the stable `F::functions()` ordering.
    // The ordering returned by `functions()` is a fixed `&'static` slice, so the
    // index is stable across encode/decode. Functions absent from that set are a
    // caller error; we fall back to index 0 (they cannot be produced by the
    // built-in generators, which only draw from `functions()`).
    fn encode_function(func: &F) -> usize {
        F::functions()
            .iter()
            .position(|candidate| candidate == func)
            .unwrap_or(0)
    }

    fn decode_function(func_idx: usize) -> Result<F, GenomeError> {
        let funcs = F::functions();
        funcs.get(func_idx).cloned().ok_or_else(|| {
            GenomeError::InvalidStructure(format!(
                "Function index {} out of range ({} functions available)",
                func_idx,
                funcs.len()
            ))
        })
    }
}

impl<T: Terminal, F: Function> fmt::Display for TreeGenome<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_sexpr())
    }
}

/// Trait for tree genome types (marker trait for operators)
pub trait TreeGenomeType: EvolutionaryGenome {
    /// The terminal type
    type Term: Terminal;
    /// The function type
    type Func: Function;

    /// Get the root of the tree
    fn root(&self) -> &TreeNode<Self::Term, Self::Func>;

    /// Get a mutable reference to the root
    fn root_mut(&mut self) -> &mut TreeNode<Self::Term, Self::Func>;

    /// Get the maximum depth
    fn max_depth(&self) -> usize;

    /// Create a new tree from a root node
    fn from_root(root: TreeNode<Self::Term, Self::Func>, max_depth: usize) -> Self;
}

impl<T: Terminal, F: Function> TreeGenomeType for TreeGenome<T, F> {
    type Term = T;
    type Func = F;

    fn root(&self) -> &TreeNode<T, F> {
        &self.root
    }

    fn root_mut(&mut self) -> &mut TreeNode<T, F> {
        &mut self.root
    }

    fn max_depth(&self) -> usize {
        self.max_depth
    }

    fn from_root(root: TreeNode<T, F>, max_depth: usize) -> Self {
        Self { root, max_depth }
    }
}

/// Free a tree node and all of its descendants without deep recursion.
///
/// Rust's compiler-generated drop glue for [`TreeNode`] is recursive (one stack
/// frame per level), so dropping a very deep tree implicitly can overflow the
/// stack. A stack-safe `Drop` impl is not possible here (see the module note on
/// deep trees), so this helper tears the tree down with an explicit work stack
/// instead. It is only able to move children out of each node *because*
/// `TreeNode` deliberately does not implement `Drop`.
pub fn drop_node_iteratively<T: Terminal, F: Function>(node: TreeNode<T, F>) {
    let mut stack = vec![node];
    while let Some(current) = stack.pop() {
        if let TreeNode::Function(_, mut children) = current {
            // Move the children onto the work stack so `current` drops shallowly
            // (its `children` vec is now empty), then process them iteratively.
            stack.append(&mut children);
        }
        // Terminal nodes (and the now-emptied function node) drop in O(1) here.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_node_terminal() {
        let node: TreeNode<ArithmeticTerminal, ArithmeticFunction> =
            TreeNode::terminal(ArithmeticTerminal::Variable(0));
        assert!(node.is_terminal());
        assert!(!node.is_function());
        assert_eq!(node.depth(), 1);
        assert_eq!(node.size(), 1);
    }

    #[test]
    fn test_tree_node_function() {
        let left = TreeNode::terminal(ArithmeticTerminal::Variable(0));
        let right = TreeNode::terminal(ArithmeticTerminal::Constant(1.0));
        let node = TreeNode::function(ArithmeticFunction::Add, vec![left, right]);

        assert!(!node.is_terminal());
        assert!(node.is_function());
        assert_eq!(node.depth(), 2);
        assert_eq!(node.size(), 3);
    }

    #[test]
    fn test_tree_node_positions() {
        // Create: (+ x0 (* 1.0 x1))
        let x0 = TreeNode::terminal(ArithmeticTerminal::Variable(0));
        let c1 = TreeNode::terminal(ArithmeticTerminal::Constant(1.0));
        let x1 = TreeNode::terminal(ArithmeticTerminal::Variable(1));
        let mul = TreeNode::function(ArithmeticFunction::Mul, vec![c1, x1]);
        let add = TreeNode::function(ArithmeticFunction::Add, vec![x0, mul]);

        let positions = add.positions();
        assert_eq!(positions.len(), 5); // root, left, right, right-left, right-right
        assert!(positions.contains(&vec![])); // root
        assert!(positions.contains(&vec![0])); // left child (x0)
        assert!(positions.contains(&vec![1])); // right child (mul)
        assert!(positions.contains(&vec![1, 0])); // mul's left child
        assert!(positions.contains(&vec![1, 1])); // mul's right child
    }

    #[test]
    fn test_tree_node_get_subtree() {
        let x0 = TreeNode::terminal(ArithmeticTerminal::Variable(0));
        let c1 = TreeNode::terminal(ArithmeticTerminal::Constant(1.0));
        let add: TreeNode<ArithmeticTerminal, ArithmeticFunction> =
            TreeNode::function(ArithmeticFunction::Add, vec![x0.clone(), c1]);

        assert_eq!(add.get_subtree(&[0]), Some(&x0));
        assert!(add.get_subtree(&[2]).is_none());
    }

    #[test]
    fn test_tree_genome_evaluate() {
        // Create: (+ x0 x1)
        let x0 = TreeNode::terminal(ArithmeticTerminal::Variable(0));
        let x1 = TreeNode::terminal(ArithmeticTerminal::Variable(1));
        let add = TreeNode::function(ArithmeticFunction::Add, vec![x0, x1]);
        let tree = TreeGenome::new(add, 5);

        assert_eq!(tree.evaluate(&[3.0, 4.0]), 7.0);
    }

    #[test]
    fn test_tree_genome_evaluate_complex() {
        // Create: (* (+ x0 1) x1) = (x0 + 1) * x1
        let x0 = TreeNode::terminal(ArithmeticTerminal::Variable(0));
        let c1 = TreeNode::terminal(ArithmeticTerminal::Constant(1.0));
        let x1 = TreeNode::terminal(ArithmeticTerminal::Variable(1));
        let add = TreeNode::function(ArithmeticFunction::Add, vec![x0, c1]);
        let mul = TreeNode::function(ArithmeticFunction::Mul, vec![add, x1]);
        let tree = TreeGenome::new(mul, 5);

        assert_eq!(tree.evaluate(&[2.0, 3.0]), 9.0); // (2 + 1) * 3 = 9
    }

    #[test]
    fn test_tree_genome_generate_full() {
        let mut rng = rand::thread_rng();
        let tree: TreeGenome<ArithmeticTerminal, ArithmeticFunction> =
            TreeGenome::generate_full(&mut rng, 3, 5);

        // Full tree with target depth 3 creates: Function -> Function -> Function -> Terminal
        // Which has depth 4 (counting levels from root to leaf)
        assert!(tree.depth() >= 3);
        assert!(tree.size() >= 1);
    }

    #[test]
    fn test_tree_genome_generate_grow() {
        let mut rng = rand::thread_rng();
        let tree: TreeGenome<ArithmeticTerminal, ArithmeticFunction> =
            TreeGenome::generate_grow(&mut rng, 5, 0.3);

        // Grow can create trees up to max_depth + 1 levels (due to counting from 0)
        assert!(tree.depth() <= 6);
        assert!(tree.size() >= 1);
    }

    #[test]
    fn test_tree_genome_to_sexpr() {
        let x0 = TreeNode::terminal(ArithmeticTerminal::Variable(0));
        let c1 = TreeNode::terminal(ArithmeticTerminal::Constant(1.0));
        let add: TreeNode<ArithmeticTerminal, ArithmeticFunction> =
            TreeNode::function(ArithmeticFunction::Add, vec![x0, c1]);
        let tree = TreeGenome::new(add, 5);

        let sexpr = tree.to_sexpr();
        assert!(sexpr.contains('+'));
        assert!(sexpr.contains("x0"));
        assert!(sexpr.contains("1.0"));
    }

    #[test]
    fn test_tree_genome_trace_roundtrip() {
        // regression: EV-04 — from_trace(to_trace(g)) must reproduce g *exactly*
        // (function identity and terminal values), not fabricate Add nodes and
        // fresh random terminals as the previous implementation did.
        let x0 = TreeNode::terminal(ArithmeticTerminal::Variable(0));
        let c1 = TreeNode::terminal(ArithmeticTerminal::Constant(2.5));
        // Use a non-Add function and mixed terminals to expose the old data loss.
        let sub = TreeNode::function(ArithmeticFunction::Sub, vec![x0, c1]);
        let x1 = TreeNode::terminal(ArithmeticTerminal::Variable(1));
        let erc = TreeNode::terminal(ArithmeticTerminal::Erc(-0.75));
        let mul = TreeNode::function(ArithmeticFunction::Mul, vec![x1, erc]);
        let root = TreeNode::function(ArithmeticFunction::Div, vec![sub, mul]);
        let original: TreeGenome<ArithmeticTerminal, ArithmeticFunction> = TreeGenome::new(root, 5);

        let trace = original.to_trace();
        let recovered: TreeGenome<ArithmeticTerminal, ArithmeticFunction> =
            TreeGenome::from_trace(&trace).unwrap();

        // Exact structural + semantic equality (variant identity, terminal values).
        assert_eq!(original, recovered);
        assert_eq!(original.max_depth, recovered.max_depth);
        assert_eq!(original.size(), recovered.size());
        assert_eq!(recovered.to_sexpr(), original.to_sexpr());

        // Evaluation results must agree across several inputs.
        for vars in [[3.0, 4.0], [-1.0, 2.0], [0.5, -0.5]] {
            assert_eq!(recovered.evaluate(&vars), original.evaluate(&vars));
        }
    }

    #[test]
    fn test_tree_generate_with_depth_explicit() {
        // EV-94: honest constructor takes an explicit maximum depth.
        let mut rng = rand::thread_rng();
        let tree: TreeGenome<ArithmeticTerminal, ArithmeticFunction> =
            TreeGenome::generate_with_depth(&mut rng, 4);
        assert!(tree.depth() >= 1);
        assert!(tree.depth() <= 5); // ramped/grow can reach max_depth (+1 level)
                                    // Degenerate depth is clamped to at least 1 and must not panic.
        let _ =
            TreeGenome::<ArithmeticTerminal, ArithmeticFunction>::generate_with_depth(&mut rng, 0);
    }

    #[test]
    fn test_arithmetic_function_ordering_is_stable() {
        // regression: EV-04 — encode_function relies on the stable ordering of
        // F::functions(); pin that ordering so encode/decode stays consistent.
        let funcs = ArithmeticFunction::functions();
        assert_eq!(funcs[0], ArithmeticFunction::Add);
        assert_eq!(funcs[1], ArithmeticFunction::Sub);
        assert_eq!(funcs[2], ArithmeticFunction::Mul);
        assert_eq!(funcs[3], ArithmeticFunction::Div);
        // Every function decodes back to itself from its own index.
        for (idx, f) in funcs.iter().enumerate() {
            let encoded = TreeGenome::<ArithmeticTerminal, ArithmeticFunction>::encode_function(f);
            assert_eq!(encoded, idx);
            let decoded =
                TreeGenome::<ArithmeticTerminal, ArithmeticFunction>::decode_function(encoded)
                    .unwrap();
            assert_eq!(&decoded, f);
        }
    }

    #[test]
    fn test_arithmetic_terminal_encode_decode_roundtrip() {
        // regression: EV-04 — terminal encode/decode must be exact for every
        // variant, including distinguishing Constant from Erc.
        for t in [
            ArithmeticTerminal::Variable(0),
            ArithmeticTerminal::Variable(7),
            ArithmeticTerminal::Constant(3.25),
            ArithmeticTerminal::Constant(-100.5),
            ArithmeticTerminal::Erc(0.0),
            ArithmeticTerminal::Erc(-0.75),
        ] {
            let (ty, val) = t.encode();
            assert_eq!(ArithmeticTerminal::decode(ty, val), t);
        }
    }

    #[test]
    fn test_tree_deep_no_stack_overflow() {
        // regression: EV-60 — a ~100k-deep degenerate tree must evaluate, report
        // depth/size, and be torn down without overflowing the call stack. The
        // previous recursive eval/depth/size and the implicit recursive drop
        // would all overflow at this depth.
        let depth = 100_000usize;
        // Build bottom-up in a loop (no recursion during construction).
        let mut root: TreeNode<ArithmeticTerminal, ArithmeticFunction> =
            TreeNode::terminal(ArithmeticTerminal::Constant(1.0));
        for _ in 0..depth {
            root = TreeNode::function(ArithmeticFunction::Neg, vec![root]);
        }
        let tree = TreeGenome::new(root, depth + 1);

        // Iterative traversals must not overflow.
        assert_eq!(tree.size(), depth + 1);
        assert_eq!(tree.depth(), depth + 1);
        // Neg applied an even number of times to 1.0 yields +1.0.
        let value = tree.evaluate(&[]);
        assert!(value.is_finite());
        assert_eq!(value, 1.0);

        // Iterative teardown must not overflow (implicit drop would recurse).
        tree.dismantle();
    }

    #[test]
    fn test_drop_node_iteratively_frees_deep_tree() {
        // regression: EV-60 — the standalone iterative teardown handles a bare
        // deep TreeNode (not wrapped in a TreeGenome) without recursion.
        let mut node: TreeNode<ArithmeticTerminal, ArithmeticFunction> =
            TreeNode::terminal(ArithmeticTerminal::Constant(0.0));
        for _ in 0..100_000 {
            node = TreeNode::function(ArithmeticFunction::Abs, vec![node]);
        }
        drop_node_iteratively(node);
    }

    #[test]
    fn test_tree_node_replace_subtree() {
        let x0 = TreeNode::terminal(ArithmeticTerminal::Variable(0));
        let x1 = TreeNode::terminal(ArithmeticTerminal::Variable(1));
        let mut add: TreeNode<ArithmeticTerminal, ArithmeticFunction> =
            TreeNode::function(ArithmeticFunction::Add, vec![x0, x1]);

        let c5 = TreeNode::terminal(ArithmeticTerminal::Constant(5.0));
        add.replace_subtree(&[0], c5);

        // Now tree should be (+ 5.0 x1)
        let tree = TreeGenome::new(add, 5);
        assert_eq!(tree.evaluate(&[0.0, 3.0]), 8.0); // 5 + 3 = 8
    }

    #[test]
    fn test_arithmetic_function_protected_div() {
        assert_eq!(ArithmeticFunction::Div.apply(&[1.0, 0.0]), 1.0);
        assert_eq!(ArithmeticFunction::Div.apply(&[6.0, 2.0]), 3.0);
    }

    #[test]
    fn test_arithmetic_function_protected_log() {
        assert_eq!(ArithmeticFunction::Log.apply(&[-1.0]), 0.0);
        assert!((ArithmeticFunction::Log.apply(&[std::f64::consts::E]) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_arithmetic_function_protected_sqrt() {
        assert_eq!(ArithmeticFunction::Sqrt.apply(&[4.0]), 2.0);
        assert_eq!(ArithmeticFunction::Sqrt.apply(&[-4.0]), 2.0); // Protected
    }

    #[test]
    fn test_tree_genome_evolutionary_genome_trait() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.0, 5);
        let tree: TreeGenome<ArithmeticTerminal, ArithmeticFunction> =
            TreeGenome::generate(&mut rng, &bounds);

        assert!(tree.dimension() >= 1);
        let decoded = tree.decode();
        assert_eq!(decoded.size(), tree.size());
    }

    #[test]
    fn test_tree_terminal_and_function_positions() {
        // Create: (+ x0 (* 1.0 x1))
        let x0 = TreeNode::terminal(ArithmeticTerminal::Variable(0));
        let c1 = TreeNode::terminal(ArithmeticTerminal::Constant(1.0));
        let x1 = TreeNode::terminal(ArithmeticTerminal::Variable(1));
        let mul = TreeNode::function(ArithmeticFunction::Mul, vec![c1, x1]);
        let add: TreeNode<ArithmeticTerminal, ArithmeticFunction> =
            TreeNode::function(ArithmeticFunction::Add, vec![x0, mul]);

        let terminal_positions = add.terminal_positions();
        assert_eq!(terminal_positions.len(), 3); // x0, 1.0, x1

        let function_positions = add.function_positions();
        assert_eq!(function_positions.len(), 2); // add, mul
    }

    #[test]
    fn test_tree_genome_display() {
        let x0 = TreeNode::terminal(ArithmeticTerminal::Variable(0));
        let c1 = TreeNode::terminal(ArithmeticTerminal::Constant(1.0));
        let add: TreeNode<ArithmeticTerminal, ArithmeticFunction> =
            TreeNode::function(ArithmeticFunction::Add, vec![x0, c1]);
        let tree = TreeGenome::new(add, 5);

        let display = format!("{}", tree);
        assert!(!display.is_empty());
    }
}
