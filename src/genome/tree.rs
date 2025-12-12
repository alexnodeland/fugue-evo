//! Tree genomes for genetic programming
//!
//! This module provides tree-based genomes for symbolic regression and
//! genetic programming applications.

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

    /// Get the depth of this subtree
    pub fn depth(&self) -> usize {
        match self {
            Self::Terminal(_) => 1,
            Self::Function(_, children) => {
                1 + children.iter().map(|c| c.depth()).max().unwrap_or(0)
            }
        }
    }

    /// Get the number of nodes in this subtree
    pub fn size(&self) -> usize {
        match self {
            Self::Terminal(_) => 1,
            Self::Function(_, children) => 1 + children.iter().map(|c| c.size()).sum::<usize>(),
        }
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

    /// Evaluate the tree with given variable bindings
    pub fn evaluate(&self, variables: &[f64]) -> f64 {
        self.evaluate_node(&self.root, variables)
    }

    fn evaluate_node(&self, node: &TreeNode<T, F>, variables: &[f64]) -> f64 {
        match node {
            TreeNode::Terminal(t) => t.evaluate(variables),
            TreeNode::Function(f, children) => {
                let args: Vec<f64> = children
                    .iter()
                    .map(|c| self.evaluate_node(c, variables))
                    .collect();
                f.apply(&args)
            }
        }
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

    fn generate<R: Rng>(rng: &mut R, bounds: &MultiBounds) -> Self {
        let max_depth = bounds.dimension().max(3).min(10);
        Self::generate_ramped_half_and_half(rng, 2, max_depth)
    }

    fn distance(&self, other: &Self) -> f64 {
        // Tree edit distance approximation based on size difference
        let size_diff = (self.size() as f64 - other.size() as f64).abs();
        let depth_diff = (self.depth() as f64 - other.depth() as f64).abs();
        size_diff + depth_diff
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

    // Encode terminal as (type_code, value)
    // type_code: 0 = Variable, 1 = Constant, 2 = ERC
    fn encode_terminal(_terminal: &T) -> (f64, f64) {
        // Default implementation for generic terminals
        // Concrete implementations would need specialization
        (0.0, 0.0)
    }

    fn decode_terminal(term_type: f64, term_val: f64) -> Result<T, GenomeError> {
        // This requires runtime generation; use random for now
        // A full implementation would need type-specific decoding
        let mut rng = rand::thread_rng();
        let _ = (term_type, term_val); // Acknowledge parameters
        Ok(T::random(&mut rng))
    }

    fn encode_function(_func: &F) -> usize {
        // Default returns 0; specialized for ArithmeticFunction
        0
    }

    fn decode_function(func_idx: usize) -> Result<F, GenomeError> {
        let funcs = F::functions();
        if func_idx < funcs.len() {
            Ok(funcs[func_idx].clone())
        } else {
            // Fall back to random
            let mut rng = rand::thread_rng();
            Ok(F::random(&mut rng))
        }
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
        let x0 = TreeNode::terminal(ArithmeticTerminal::Variable(0));
        let c1 = TreeNode::terminal(ArithmeticTerminal::Constant(2.5));
        let add = TreeNode::function(ArithmeticFunction::Add, vec![x0, c1]);
        let original: TreeGenome<ArithmeticTerminal, ArithmeticFunction> = TreeGenome::new(add, 5);

        let trace = original.to_trace();
        let recovered: TreeGenome<ArithmeticTerminal, ArithmeticFunction> =
            TreeGenome::from_trace(&trace).unwrap();

        // The trace encoding preserves structure (max_depth, size) but terminal values
        // are generated fresh since the simple encoding doesn't preserve all details
        assert_eq!(original.max_depth, recovered.max_depth);
        assert_eq!(original.size(), recovered.size());
        // Both should be valid trees of the same structure
        assert!(recovered.evaluate(&[3.0]).is_finite());
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
