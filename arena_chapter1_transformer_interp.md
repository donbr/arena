## Chapter 1: Transformer Interpretability - Unmasking the Black Box of Language Models

This chapter plunges into the world of transformers, the dominant architecture in natural language processing, and tackles the challenge of understanding their internal workings through the lens of mechanistic interpretability. Building on the deep learning foundations established in Chapter 0, it guides you through a comprehensive exploration of transformer architecture, interpretability tools, and advanced concepts, culminating in hands-on analysis and even manipulating model behavior.

**Section 1: Transformer from Scratch - Building the Beast**

This section throws you into the deep end, tasking you with building a GPT-2 style transformer from scratch using just PyTorch's tensor operations. This hands-on approach fosters a deep understanding of transformer internals and their inner workings. 

* **Inputs & Outputs:**
    * Explores the purpose of a transformer: to model text and generate predictions for the next token.
    * Explains the difference between causal attention (used in GPT-like models, restricting information flow to forward direction) and bidirectional attention (allowing information flow in both directions).
    * Reviews tokenization - converting text to integer tokens for model input – and logits – the unnormalized scores representing the model's probability distribution over tokens. 
* **Clean Transformer Implementation:**
    * Introduces the core components of a transformer, each acting on the residual stream: 
        * **LayerNorm:**  Normalizes input vectors to have zero mean and unit variance.
        * **Positional embedding:** Provides positional information to the model, allowing it to distinguish between tokens at different positions.
        * **Attention:** Calculates attention patterns, determining how information is moved between different positions in the sequence.
        * **MLP:** Performs non-linear transformations on the residual stream using linear layers and activation functions.
        * **Embedding:** Maps tokens to initial residual stream vectors.
        * **Unembedding:** Converts final residual stream vectors into a probability distribution over tokens.
    * You'll implement each of these modules from scratch, step by step, culminating in assembling a full transformer model. 
    * The section emphasizes the concept of a residual stream and how information flows through the different components.
* **Training a Transformer:**
    * Explains how to train a transformer using a dataset of text and the cross-entropy loss function.
    * You'll write a basic training loop and observe how the loss decreases as the model learns to predict the next token.
    * The section discusses interpreting the loss curve, relating it to the types of patterns the model learns (e.g., word frequencies, bigrams).
* **Sampling from a Transformer:**
    * Explains how to generate text by sampling from the model's output probability distribution.
    * Covers various sampling methods, including greedy search, top-k sampling, and beam search.
    * Introduces caching - storing intermediate activations to speed up text generation.
    * You'll implement sampling functions and (optionally) adapt them to use caching.

**Section 2: Intro to Mech Interp - Opening the Black Box**

This section dives into the exciting world of mechanistic interpretability, aiming to understand *how* transformers learn to solve tasks, not just *that* they can solve them. You'll be introduced to powerful tools and concepts for analyzing model behavior.

* **TransformerLens: Introduction**
    * Introduces the TransformerLens library, specifically designed for performing mechanistic interpretability research on GPT-2 style language models.
    * Covers loading pre-trained models, understanding the model architecture (similar to the architecture you built from scratch), and using the tokenizer.
    * Explains how to use `run_with_cache` to save model activations and access them for analysis.
    * Shows how to visualize attention patterns using the `circuitsvis` library.
* **Finding Induction Heads**
    * Introduces the concept of **induction heads** - circuits in transformers that perform a simple form of in-context learning, generalizing from one observed pattern to others.
    * Shows how to identify these heads through their characteristic attention pattern on repeating sequences.
    * You'll write functions to detect different types of attention heads based on their attention patterns.
* **TransformerLens: Hooks**
    * Introduces **hooks** - functions that allow you to access and intervene on activations within the model during forward passes.
    * Covers using hooks to extract activations, process them, and store them externally.
    * Explains how to use hooks to perform **ablation** - removing the contribution of specific components to analyze their importance.
    * You'll build tools to perform **logit attribution**, identifying which components contribute most to the model's output for a specific task.
* **Reverse-engineering Induction Circuits**
    * Explores reverse-engineering circuits by directly inspecting model weights, considered a "gold standard" in interpretability.
    * Covers analyzing QK (query-key) and OV (output-value) circuits by multiplying through matrices, aided by the `FactoredMatrix` class.
    * You'll analyze composition scores to identify potential connections between heads and perform targeted ablations to verify their role in the circuit.

**Section 3: Indirect Object Identification (IOI) - A Real-World Circuit**

This section focuses on dissecting a specific circuit in GPT-2 Small: the **indirect object identification (IOI) circuit**, responsible for correctly completing sentences like *"When Mary and John went to the store, John gave a drink to..."*. It utilizes the IOI task as a case study to demonstrate a range of interpretability techniques.

* **Model & Task Setup:**
    * Explains the IOI task in detail, its benefits as an interpretability target, and how to measure the model's performance using the **logit difference** metric.
    * Encourages you to brainstorm how this behavior might be implemented within a transformer before diving into experiments.
* **Logit Attribution:**
    * Performs direct logit attribution to identify heads that significantly contribute to the logit difference, revealing potential key components of the circuit.
    * Introduces the `cache.decompose_resid` and `cache.stack_head_results` methods for decomposing the residual stream and analyzing the contributions of individual layers and heads.
* **Activation Patching:**
    * Introduces **activation patching** - replacing an activation with a corrupted version (e.g., from a different input) to analyze its causal impact on the model's output.
    * You'll implement activation patching using hooks and use it to identify key layers and sequence positions where crucial information is stored and processed.
* **Path Patching:**
    * Introduces **path patching** - a more refined form of activation patching that analyzes the importance of specific paths between components.
    * You'll implement path patching using hooks and use it to replicate specific findings from the IOI paper, providing more rigorous evidence for the circuit.
* **Paper Replication:**
    * Guides you through replicating other key results from the IOI paper, including:
        * Analyzing the behavior of specific head types within the circuit (duplicate token heads, S-inhibition heads, name mover heads).
        * Performing a complex ablation experiment that removes all components *except* those identified as part of the circuit and verifying performance recovery.
* **Bonus:**
    * Suggests additional exercises for deeper exploration, such as:
        * Investigating negative name mover heads (heads that seem to contribute negatively to the task).
        * Analyzing the role of induction heads within the IOI circuit.
        * Finding adversarial examples that exploit the model's limitations.
        * Thinking about the criteria of **faithfulness**, **completeness**, and **minimality** when evaluating an interpretability hypothesis.

**Section 4: Superposition & Sparse Autoencoders (SAEs) - Unraveling Complex Representations**

This section tackles the challenging concept of **superposition**, where a model represents more features than it has dimensions. It introduces **sparse autoencoders (SAEs)** as a potential tool for disentangling these representations.

* **Toy Models of Superposition: Basics:**
    * Introduces Anthropic's toy models of superposition, highlighting the trade-off between feature benefit (representing more features) and interference (representing features non-orthogonally).
    * You'll build and train the toy model, replicate key results, and visualize superposition in 2D and higher dimensions using various techniques (bar graphs, heatmaps, scatter plots).
    * Explores how superposition varies with feature importance, sparsity, and correlations (positive/negative).
* **Toy Models of Superposition: Extensions:**
    * Takes deeper dives into specific aspects of the toy models paper:
        * **Feature Geometry:**  Investigates the geometric arrangements of features in superposition, highlighting how they organize into specific structures like pentagons and tetrahedrons.
        * **Double Descent & Superposition:** Explores the connection between superposition and the phenomenon of deep double descent, where test error decreases after initially increasing with model complexity. 
* **Sparse Autoencoders:**
    * Introduces Sparse Autoencoders (SAEs) - networks trained to compress data into a sparse representation, potentially helping to separate superposed features.
    * You'll train SAEs on the toy models from previous sections, analyze their reconstruction performance, and visualize how they learn to identify and represent individual features.
    * Explores SAEs trained on real language models, observing their ability to uncover interpretable features and investigating the phenomenon of shrinkage.

**Section 5: Function Vectors & Model Steering - Manipulating Model Behavior**

This section shifts from interpreting existing model behavior to exploring whether we can actively steer a model to behave differently through interventions on its internal activations.

* **Introduction:**
    * Introduces the concept of **function vectors** - vectors extracted from models trained on in-context learning (ICL) tasks that can be used to trigger specific behaviors in zero-shot prompts.
    * Presents the `nnsight` library, designed to facilitate this kind of work (and other interpretability research) on very large language models (e.g., GPT-J-6B).
    * Reviews key concepts like tokenization, model outputs, and the process of running forward passes with `nnsight`, including caching and saving activations.
* **Task-encoding Hidden States:**
    * Poses the question of whether hidden states in a transformer can encode specific tasks learned during ICL.
    * You'll extract a "task-encoding" vector from a set of prompts for the antonym task and use it to intervene on the model's residual stream, inducing antonym generation from zero-shot prompts.
* **Function Vectors:**
    * Replicates the main findings of the function vectors paper, demonstrating how to:
        * Extract function vectors from ICL tasks.
        * Identify key attention heads that contribute to the task.
        * Patch with these vectors to induce task-solving behavior on randomly shuffled prompts.
    * You'll experiment with steering the model on different tasks (e.g., country-capitals) and explore techniques for multi-token generation with `nnsight`.
* **Steering Vectors in GPT2-XL:**
    * Introduces Alex Turner's work on **steering vectors**, which aims to induce broader behavioral changes in models (e.g., shifting sentiment) rather than specific task-solving behavior.
    * You'll replicate the results of their initial post, showing how adding steering vectors to the residual stream can alter the model's completions.
* **Bonus:**
    * Discusses various extensions to this research, including:
        * Further exploration of results from the function vectors paper (decoded vocabulary, vector algebra).
        * Replicating results from related papers on inference-time intervention and steering Llama 2.
        * Brainstorming potential capstone projects and research directions in this area.

Throughout this chapter, you'll be exposed to a wide range of interpretability techniques, tools, and concepts, progressing from simply observing attention patterns to actively manipulating model behavior. By the end, you'll not only have a deeper understanding of how transformers work but also the skills to start probing and influencing their inner workings, opening up new avenues for AI alignment research.
