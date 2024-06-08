##  Unveiling the Secrets of Transformers: A Deep Dive into Chapter 1 Exercises

The `chapter1_transformer_interp/exercises` folder of the ARENA 3.0 repository is a journey into the heart of transformer models, equipping you with the tools and insights to understand their intricate workings and even influence their behavior. This section builds upon the fundamental concepts from Chapter 0, delving into the complexities of attention mechanisms, circuit analysis, and advanced interpretability techniques.

**Part 1: Transformer from Scratch (`part1_transformer_from_scratch`)**

This section revisits the task of building a transformer, but this time with a focus on clarity and compatibility with the TransformerLens library, which you'll be using extensively for the rest of the chapter.

* **File:** `solutions.py`
    * Contains complete implementations of the transformer modules and training/sampling functions.
* **File:** `tests.py`
    * Provides test functions to verify your implementations.

**Key Exercises:**

1. **Implementing Transformer Modules:**
    * You'll re-implement core transformer modules, including:
        * **Embedding:** Maps input tokens to vector representations.
        * **Positional Embedding:** Encodes positional information, allowing the model to distinguish between tokens at different positions.
        * **Attention:** The heart of the transformer, computing attention patterns to determine how information flows between tokens.
        * **Multi-Head Attention:** Extends attention by allowing multiple independent attention heads to operate in parallel.
        * **LayerNorm:**  Normalizes activations to stabilize training.
        * **MLP:**  Performs non-linear transformations on the residual stream.
        * **Transformer Block:** Combines all the above modules into a single transformer layer.
        * **DemoTransformer:**  Assembles multiple transformer blocks into a complete transformer model.
        * **Unembedding:**  Maps the final residual stream to a probability distribution over tokens.
    * Each exercise focuses on understanding the functionality of these modules, their mathematical implementations, and how they contribute to the overall transformer architecture.
    * You'll also learn how to initialize weights and biases appropriately, following conventions used in TransformerLens.
2. **Causal Attention Mask:**
    * You'll implement the `apply_causal_mask` function, which masks attention scores to ensure that information only flows forward in the sequence (preventing the model from "cheating" by looking at future tokens).
3. **Training Loop:**
    * You'll write a training loop to train your transformer on a text dataset using the cross-entropy loss function and an optimizer.
    * The exercise emphasizes understanding the steps involved in a training iteration (forward pass, loss calculation, backpropagation, parameter update) and observing how the loss decreases as the model learns.
4. **Sampling from the Model:**
    * You'll implement functions to generate text by sampling from the model's output probability distribution, including:
        * **Greedy Sampling:**  Choosing the most probable token at each step.
        * **Temperature Sampling:**  Scaling the logits to adjust the randomness of the generated text.
        * **Top-k Sampling:**  Sampling from the top k most probable tokens.
        * **Nucleus (Top-p) Sampling:**  Sampling from tokens whose cumulative probability exceeds a threshold p.
        * **Beam Search:**  Exploring multiple candidate sequences in parallel to find the most likely completion.
    * You'll also learn about caching activations to speed up text generation, and optionally adapt your sampling functions to use caching.

**Part 2: Introduction to Mechanistic Interpretability (`part2_intro_to_mech_interp`)**

This section formally introduces **mechanistic interpretability** - the art and science of understanding the algorithms learned by deep learning models. It guides you through using the TransformerLens library to analyze model behavior, focusing on identifying and understanding specific circuits within transformers.

* **File:** `solutions.py`
    * Contains solutions to the exercises.
* **File:** `tests.py`
    * Provides test functions to verify your implementations.

**Key Exercises:**

1. **TransformerLens Basics:**
    * You'll load a pre-trained `HookedTransformer` model (a variant of GPT-2 designed for interpretability) and analyze its configuration using the `cfg` attribute.
    * You'll then run the model on a sample text, exploring the different return types (`logits`, `loss`, `both`, `None`) and observing the model's output.
    * You'll learn about the model's architecture and how to access various components and activations using the cache.
    * The section also reviews using the tokenizer to convert between text and tokens and introduces the `circuitsvis` library for visualizing attention patterns.
2. **Finding Induction Heads:**
    * You'll delve into the concept of **induction heads** - circuits that enable transformers to perform in-context learning, generalizing from one observed pattern to others.
    * You'll analyze attention patterns on a repeating sequence of random tokens, observing the characteristic "stripe" pattern that indicates an induction head.
    * You'll then write functions to automatically detect induction heads based on their attention patterns.
3. **TransformerLens: Hooks:**
    * This section introduces **hooks** - a powerful feature in TransformerLens that allows you to access and intervene on activations within the model during forward passes.
    * You'll learn about adding hooks to specific activations and running the model with hooks to extract and process intermediate values.
    * You'll also use hooks to perform **ablation** - removing the contribution of specific components to analyze their impact on the model's output.
    * Finally, you'll implement functions to perform **logit attribution**, identifying the components that contribute most to the logit difference (a metric introduced in the previous section) for a specific task.
4. **Reverse-engineering Induction Circuits:**
    * This section explores how to **reverse-engineer circuits** by directly analyzing the model's weights.
    * You'll learn about **QK (query-key)** and **OV (output-value) circuits**, which represent the transformations applied to keys and queries before attention and to values after attention, respectively.
    * You'll use the `FactoredMatrix` class to efficiently analyze these circuits, examining their eigenvalues, singular value decomposition, and norms.
    * You'll also explore **composition scores** to identify potential connections between heads and perform targeted ablations to verify these connections.

**Part 3: Indirect Object Identification (`part3_indirect_object_identification`)**

This section focuses on a specific circuit in GPT-2 Small: the **indirect object identification (IOI) circuit**, responsible for correctly completing sentences like *"When Mary and John went to the store, John gave a drink to..."*. It leverages the IOI task as a case study to demonstrate a range of interpretability techniques.

* **File:** `ioi_circuit_extraction.py`
    * Contains helper functions for extracting and ablating the IOI circuit in GPT-2 Small.
* **File:** `ioi_dataset.py`
    * Defines a dataset specifically designed for evaluating the IOI task.
* **File:** `solutions.py`
    * Contains solutions to the exercises.
* **File:** `tests.py`
    * Provides test functions to verify your implementations.

**Key Exercises:**

1. **Model & Task Setup:**
    * You'll load a pre-trained GPT-2 Small model and become familiar with the IOI dataset, which contains prompts designed to elicit indirect object identification behavior.
    * You'll implement a function to evaluate the model's performance on the IOI task using the **logit difference** metric - the difference in logit between the correct and incorrect indirect object.
2. **Logit Attribution:**
    * You'll perform **direct logit attribution**, analyzing which heads contribute most significantly to the final logit difference.
    * You'll use TransformerLens's `cache.decompose_resid` and `cache.stack_head_results` methods to decompose the residual stream and analyze the contributions of individual layers and heads to the logit difference.
    * This will reveal that specific heads in later layers are primarily responsible for the IOI behavior.
3. **Activation Patching:**
    * This section introduces **activation patching**, a technique for assessing the causal impact of specific activations on the model's output. You'll implement this technique using hooks, replacing a chosen activation with a corrupted version (e.g., from a different input).
    * You'll use activation patching to investigate where in the residual stream crucial information for the IOI task is stored and processed, focusing on the layers and sequence positions identified as important in the previous logit attribution step.
    * By patching with activations from different input sequences, you'll start to build a picture of the model's internal algorithm for identifying indirect objects.
4. **Path Patching:**
    * You'll implement **path patching**, a more refined form of activation patching that focuses on analyzing the importance of specific paths between model components.
    * This technique allows you to test more specific hypotheses about the flow of information within the model, pinpointing the causal relationships between different activations.
    * You'll use path patching to replicate several key results from the IOI paper, providing more rigorous evidence for the specific pathways involved in the IOI circuit.
5. **Paper Replication:**
    * This section guides you through replicating the main findings of the IOI paper, including:
        * Identifying and analyzing the behavior of specific types of heads within the IOI circuit: **duplicate token heads**, **S-inhibition heads**, and **name mover heads**.
        * Performing a complex **ablation experiment** that removes all components *except* those identified as part of the IOI circuit and verifying that the model still exhibits the IOI behavior, providing strong evidence for the circuit's completeness and faithfulness.
* **Bonus:**
    * This section offers suggestions for further explorations and more challenging exercises, including:
        * Investigating the role of **negative name mover heads** - heads that seem to contribute negatively to the IOI task – and understanding how they might help the model avoid overconfidence and generalize better.
        * Analyzing the interplay between the IOI circuit and **induction heads**, shedding light on the relationship between different types of circuits within the model.
        * Using your understanding of the circuit to craft **adversarial examples** - inputs designed to fool the model and expose its limitations.
        * Reflecting on the criteria of **faithfulness, completeness, and minimality** when evaluating the validity of an interpretability hypothesis.

**Part 4: Superposition & Sparse Autoencoders (`part4_superposition_and_saes`)**

This section delves into the fascinating and challenging concept of **superposition**, where a model represents more features than its dimensions allow, cramming multiple pieces of information into the same activation space. It also introduces **Sparse Autoencoders (SAEs)** as a tool to help disentangle these complex representations.

* **Files:**
    * `solutions.py`: Solutions to the exercises.
    * `tests.py`: Contains test functions to verify your implementations.
    * `utils.py`: Provides helper functions for visualizing feature representations, superposition, and sparsity.

**Key Exercises:**

1. **Toy Models of Superposition: Basics:**
    * You'll build and train a toy model of superposition based on Anthropic's research. This model consists of a linear map from a higher-dimensional feature space to a lower-dimensional "bottleneck" space and then back to the original feature space.
    * The exercises introduce the concepts of **feature importance** and **sparsity** and explore how they affect the model's behavior and the emergence of superposition.
    * You'll visualize the learned representations in 2D and higher dimensions using techniques like heatmaps, bar graphs, and scatter plots, observing how features organize themselves in the bottleneck space and how superposition allows the model to represent more features than dimensions.
2. **Toy Models of Superposition: Extensions:**
    * This section explores two deeper aspects of superposition:
        * **Feature Geometry:**  You'll investigate the geometric arrangements of features in superposition, discovering how they tend to organize into specific structures like pentagons and tetrahedrons, and analyzing the concept of dimensionality to quantify the fraction of a dimension occupied by a specific feature.
        * **Double Descent & Superposition:**  You'll explore the relationship between superposition and the phenomenon of deep double descent, where test error decreases after initially increasing with model complexity. You'll investigate how the model transitions from memorizing data points to learning generalizing solutions, and how superposition plays a role in this process.
3. **Sparse Autoencoders:**
    * This section introduces **Sparse Autoencoders (SAEs)** - networks specifically designed to learn sparse representations, potentially offering a way to disentangle superposed features.
    * You'll implement and train SAEs on the toy models from previous exercises, analyzing their ability to reconstruct the original features and visualizing how they learn to identify and separate individual features.
    * You'll also explore SAEs trained on real language models, observing how they uncover interpretable features and investigating the phenomenon of shrinkage, where the sparsity objective can sometimes suppress relevant features.

**Part 5: Function Vectors & Model Steering (`part5_function_vectors_and_model_steering`)**

This section goes beyond interpreting existing model behavior and explores how to actively influence and steer a model's output by directly manipulating its internal activations.

* **Files:**
    * `data/`:  Contains data files for various in-context learning tasks, including antonym pairs and country-capital pairs.
    * `solutions.py`:  Solutions to the exercises.
    * `tests.py`:  Contains test functions to verify your implementations.

**Key Exercises:**

1. **Introduction to `nnsight`:**
    * You'll be introduced to the `nnsight` library, a tool designed to facilitate interpretability research on very large language models.
    * You'll learn the basics of running forward passes with `nnsight`, extracting and saving activations, working with the tokenizer, and understanding the model's output.
2. **Task-Encoding Hidden States:**
    * You'll explore the concept of **task-encoding hidden states**, investigating whether the residual stream in a transformer can encode the task being performed during in-context learning (ICL).
    * You'll extract a vector `h` from a set of ICL prompts for the **antonym task** and use it to intervene on the model's residual stream during inference.
    * You'll then observe how adding this vector to the residual stream induces antonym generation on zero-shot prompts, demonstrating that hidden states can indeed encode task information.
3. **Function Vectors:**
    * You'll delve deeper into the concept of function vectors, learning how to:
        * Identify **key attention heads** whose output significantly affects the model's performance on an ICL task, using metrics like **average logit difference**.
        * **Extract function vectors** from these important attention heads, representing the contribution of those heads to the task.
        * **Intervene with these vectors** on randomly shuffled or corrupted ICL prompts, demonstrating their ability to induce the desired task behavior even when the original context is disrupted.
    * You'll experiment with different ICL tasks (e.g., antonyms, country-capitals) and observe how function vectors can be used to steer the model's output.
4. **Steering Vectors in GPT2-XL:**
    * This section introduces **steering vectors**, a related concept to function vectors that focuses on inducing broader behavioral changes in models rather than specific task-solving behavior.
    * You'll replicate the results from a LessWrong post by Alex Turner et al., which demonstrates how adding pre-computed steering vectors to the residual stream can induce changes in sentiment and other aspects of the model's output.
* **Bonus:**
    * This section suggests various extensions to this research, including:
        * Exploring additional results from the function vectors paper, such as the **decoded vocabulary** of function vectors (analyzing which tokens are most strongly associated with a particular function vector) and performing **vector algebra** on function vectors to understand their compositionality.
        * Replicating results from related papers on **inference-time intervention**, exploring different techniques for finding intervention vectors and analyzing their effectiveness in eliciting truthful answers from language models.
        * Applying steering vector techniques to larger models like Llama 2 and investigating how they affect model behavior on tasks related to **myopia, power-seeking, and other high-level attributes**.

**Part 6: OthelloGPT (`part6_othellogpt`)**

This section explores a fascinating case study of emergent world representation in a transformer model trained to play the board game Othello. The model, called OthelloGPT, learns to represent the game board and play legal moves, even though it's never explicitly provided with the board state during training. 

* **Files:**
    * `othello_world.py`:  Defines an Othello environment, which can be used to play games and generate data (not a `gym` env).
    * `solutions.py`:  Contains solutions to the exercises.
    * `tests.py`:  Contains test functions to verify your implementations.

**Key Exercises:**

1. **Model Setup & Linear Probes:**
    * You'll load a pre-trained OthelloGPT model and explore its architecture, which is a simplified transformer with no MLP layers and learned positional embeddings.
    * You'll learn how to represent an Othello game board as a 2D array and how the model is trained to predict legal moves from a sequence of moves.
    * You'll also be introduced to **linear probes**, which are linear functions applied to the residual stream to extract information about the model's internal representations.
    * You'll use a pre-trained linear probe to extract the board state from the model's activations, demonstrating that the model has learned an emergent world representation of the board.
2. **Looking for Modular Circuits:**
    * You'll use the linear probe to analyze the model's circuits, investigating:
        * **Which neurons** in the model are most relevant for representing the board state.
        * **When and where** in the residual stream this information is represented.
    * You'll perform **activation patching** to test hypotheses about how the model encodes board information, manipulating specific activations to see how they affect the probe's output and the model's predicted moves.
3. **Neuron Interpretability: A Deep Dive:**
    * You'll take a deep dive into understanding a single neuron in the model, applying techniques like:
        * **Direct logit attribution:**  Analyzing how the neuron's output weights contribute to the final logits for each possible move.
        * **Max activating datasets:**  Identifying game states that strongly activate the neuron.
        * **Spectrum plots:**  Visualizing the distribution of neuron activations across a dataset, categorized by whether a specific hypothesis about the neuron's behavior is true.
    * You'll explore the limitations of these techniques and the challenges of fully understanding a neuron's function.
4. **Training a Probe:**
    * You'll learn how to train a linear probe from scratch, using a dataset of Othello games and the cross-entropy loss function.
    * You'll implement a training loop for the probe, using an optimizer to update its weights based on the model's residual stream activations and the true board states.
    * This exercise reinforces your understanding of training neural networks and provides a practical application of linear probes.
* **Bonus:**
    * This section suggests various avenues for further exploration, including:
        * Investigating the model's ability to learn the rules of Othello, such as legal move constraints and game termination conditions.
        * Analyzing the model's ability to plan ahead and predict future game states.
        * Exploring more sophisticated probing techniques, such as non-linear probes or probes that target specific components of the model.
        * Investigating the effects of different training regimes on the model's emergent world representation.

**Part 7: Balanced Bracket Classifier (`part7_balanced_bracket_classifier`)**

This section explores a transformer model trained to classify bracket strings as balanced or unbalanced. This simple yet insightful task allows for a deep analysis of the model's decision-making process, revealing how it learns to solve algorithmic problems and how interpretability techniques can be applied to understand these solutions.

* **File:** `brackets_data.json`
    * Contains a dataset of bracket strings and their corresponding balanced/unbalanced labels.
* **File:** `brackets_datasets.py`
    * Defines a dataset class and a simple tokenizer for the bracket classification task.
* **File:** `brackets_model_state_dict.pt`
    * Contains the pre-trained weights for the balanced bracket classifier model.
* **File:** `solutions.py`
    * Contains solutions to the exercises.
* **File:** `tests.py`
    * Provides test functions to verify your implementations.

**Key Exercises:**

1. **Bracket Classifier:**
    * Introduces the task of balanced bracket classification and how it can be framed as a binary classification problem for a transformer.
    * Explains the model's architecture, including its use of **bidirectional attention** (unlike the causal attention in GPT-2), **padding tokens** to handle variable sequence lengths, and **masking of padding tokens** during attention calculations.
    * You'll load a pre-trained model, tokenize example bracket strings, and evaluate the model's predictions.
    * The section encourages you to think about the algorithmic solutions that a transformer might learn for this task, given its inductive biases.
2. **Moving Backwards:**
    * This section focuses on understanding how the model arrives at its classification decision, working backwards from the output logits to identify important components.
    * Introduces the concept of **direct logit attribution**, analyzing which heads contribute most to the logit difference between the "balanced" and "unbalanced" classes.
    * You'll use TransformerLens's `cache.apply_ln_to_stack` method to apply **LayerNorm scaling** to the final residual stream, enabling a linear approximation of the contribution of each component to the logit difference.
    * You'll explore the challenges posed by LayerNorm for interpretability and learn how to handle them through techniques like folding the LayerNorm parameters into the unembedding matrix.
3. **Understanding the Total Elevation Circuit:**
    * This section delves into analyzing the model's **attention patterns** and interpreting them to understand the algorithm it has learned.
    * You'll visualize attention patterns using circuitsvis and identify recurring patterns like attending to the previous token, attending to the current token, and attending to the first token.
    * You'll focus on understanding how the model detects **total elevation failures** (cases where the total number of left and right brackets don't match) using a specific attention head (head 2.0).
    * You'll then analyze the inputs to this head, focusing on the contributions of MLP layers and understanding how they encode information about bracket tallies.
    * This exercise showcases the power of combining attention analysis with an understanding of the model's weights to decipher the model's internal logic.
* **Bonus:**
    * This section suggests several avenues for further exploration:
        * Investigating the model's mechanism for detecting **anywhere-negative failures** (cases where the cumulative count of right brackets falls below the count of left brackets at any point in the sequence).
        * Crafting **adversarial examples** that exploit the model's limitations and cause it to misclassify bracket strings, highlighting the importance of robustness analysis in interpretability.
        * Exploring other aspects of the model's behavior, such as how it handles early closing parentheses.
        * Using the insights gained from this toy model to tackle more complex algorithmic problems and replicate relevant papers.

**Part 8: Grokking & Modular Arithmetic (`part8_grokking_and_modular_arithmetic`)**

This section focuses on analyzing a transformer model trained on modular addition. It delves into the concept of **grokking** - a phenomenon where the model's performance suddenly jumps after a period of stagnation, demonstrating a sudden shift from memorization to generalization. The analysis reveals a surprising connection between the model's internal representation and the **Fourier basis**.

* **Files:**
    * `Grokking/`: This subdirectory contains the training data and pre-trained models used in this section.
    * `my_utils.py`: Provides various helper functions, including functions to work with the Fourier basis and analyze model activations.
    * `solutions.py`:  Contains solutions to the exercises.
    * `tests.py`:  Contains test functions to verify your implementations.

**Key Exercises:**

1. **Periodicity & Fourier Basis:**
    * Introduces the concept of **periodicity** in functions and data, and how it can be represented using the **Fourier basis**.
    * You'll implement functions to generate the 1D and 2D Fourier basis, understanding how they capture different frequencies of oscillation in data.
    * You'll then analyze the model's activations and observe their periodic nature, suggesting that the model might be using the Fourier basis internally.
    * You'll implement functions to perform **discrete Fourier transforms (DFTs)** on 1D and 2D data, which decompose the data into its constituent frequencies represented by the Fourier basis.
2. **Circuits & Feature Analysis:**
    * In this section, you'll leverage your understanding of the Fourier basis to dissect the model's algorithm for performing modular addition.
    * You'll analyze the model's weights in the Fourier basis, observing that they are highly **sparse**, indicating a preference for certain frequencies.
    * You'll identify **key frequencies** that are crucial for the model's performance and formulate hypotheses about how the model combines these frequencies to compute the sum of two numbers modulo p.
    * You'll test your hypotheses using various techniques, including:
        * **Projecting activations onto specific Fourier directions:**  Verifying that these directions capture most of the model's variance.
        * **Ablating specific frequencies:**  Observing the impact on the model's performance, providing evidence for their role in the algorithm.
    * By the end of this section, you should have a clear understanding of the model's algorithm and how it leverages the Fourier basis to perform modular addition.
3. **Analysis During Training:**
    * This section explores the model's **training dynamics**, analyzing how its internal representations and behavior evolve over time.
    * You'll implement the **excluded loss** metric, which measures the model's performance when specific Fourier frequencies are removed from its output. This helps demonstrate that the model learns to rely on these key frequencies for generalization, even before the grokking point.
    * You'll investigate the development of the model's embedding, observing how it learns to prioritize certain frequencies in the Fourier basis over the course of training.
    * You'll also analyze the emergence of other capabilities in the model, such as commutativity (the ability to add numbers in any order), by visualizing attention patterns and quantifying their behavior over time.
    * Finally, you'll examine the total sum of squared weights for different parameters throughout training, revealing distinct phases in the model's learning process and providing further insight into the grokking phenomenon.
* **Discussion & Future Directions:**
    * This section reflects on the key takeaways from the grokking analysis and discusses potential extensions and future research directions, including:
        * Investigating the relationship between phase changes and the development of generalizing circuits in deep learning models.
        * Exploring the role of different regularization techniques in shaping the model's learning process and the emergence of grokking.
        * Examining how interpretability tools can be used to influence training dynamics and potentially guide models towards learning more desirable algorithms.
        * Applying the insights gained from this toy model to more complex algorithmic problems and real-world language models.

By working through these exercises, you'll gain a deeper appreciation for the intricacies of transformer models, their ability to learn complex algorithms, and the power of mechanistic interpretability to unravel the secrets of their inner workings. You'll also develop a set of valuable skills for analyzing and influencing model behavior, laying the foundation for further exploration of this exciting field.

This concludes a detailed summary of the `chapter1_transformer_interp/exercises` content.
