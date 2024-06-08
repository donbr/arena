## Chapter 0: Fundamentals - A Deep Dive into the Foundations of Deep Learning

The "Fundamentals" chapter of the ARENA 3.0 program serves as a comprehensive introduction to the core concepts and practical skills necessary for diving into the world of deep learning. It's designed to equip you with the knowledge and confidence to tackle the more advanced topics of transformer interpretability and reinforcement learning later in the curriculum.

Here's a detailed breakdown of each section within this chapter:

**Section 1: Prerequisites - Setting the Stage**

This section is crucial for ensuring you have the necessary background knowledge before delving into deep learning. It covers:

* **Core Concepts & Knowledge:**
    * **Math:**  A strong grasp of linear algebra (linear transformations, matrix operations, basic properties), probability & statistics (probability rules, expected value, standard deviation), calculus (differentiation, partial differentiation, chain rule, Taylor series), and information theory (entropy, KL divergence) is essential for understanding how deep learning models work. The material provides links to external resources like 3Blue1Brown videos and articles on LessWrong to refresh your memory or learn these concepts.
    * **Programming:**  Proficiency in Python is crucial as it's the language used throughout the program. You'll need to be comfortable with basic programming constructs and common data structures.
    * **Libraries:** Familiarity with core libraries like NumPy (for numerical computation and array manipulation) and PyTorch (for building and training deep learning models) is essential. The section provides links to introductory tutorials and exercises to help you get up to speed.
* **Einops, Einsum & Tensor Manipulation:**
    * Introduces the powerful libraries `einops` (for elegant and efficient tensor manipulation) and `einsum` (for performing einstein summation convention operations).
    * Includes exercises on using these libraries to reshape, transpose, and combine tensors, making it easier to work with complex deep learning models.

**Section 2: Ray Tracing - Visualizing Batched Operations**

This section utilizes a simple graphics rendering task (ray tracing) to introduce the concept of batched operations in PyTorch. 

* **1D Image Rendering:**
    * You'll start by defining rays in 2D space and learn how to parameterize and visualize them using the `plotly` library.
    * You'll then implement a function to generate a batch of rays, setting the stage for batched operations.
* **Ray-Object Intersection:**
    * You'll learn how to determine if a ray intersects a line segment (representing a simple object in the scene) by solving linear equations.
    * You'll implement functions to check for intersection and handle cases where solutions don't exist (e.g., parallel lines).
    * The section introduces typechecking using the `jaxtyping` and `typeguard` libraries to improve code clarity and catch errors.
* **Batched Ray-Segment Intersection:**
    * Extends the previous concept to handle multiple rays and line segments simultaneously, introducing **broadcasting** and **logical reductions**.
    * You'll implement a batched version of the intersection function using `einops.repeat` to efficiently handle all ray-segment combinations.
* **2D Rays:**
    * Moves from 2D to 3D space, defining rays originating from the camera and passing through a 2D screen.
    * You'll implement a function to generate these 3D rays.
* **Triangle Coordinates & Intersection:**
    * Introduces the concept of **barycentric coordinates** to represent points within a triangle and how to solve for their intersection with rays.
    * You'll implement a function to determine if a ray intersects a triangle.
* **Single-Triangle Rendering:**
    * Applies the intersection algorithm to render a single triangle as a 2D image, introducing the idea of **views** and **copies** in PyTorch.
    * The section covers essential concepts like:
        * `storage` objects to understand the underlying data layout in memory.
        * `Tensor._base` to access the original tensor from a view.
    * You'll implement a function to raytrace a single triangle and visualize the results.
* **Mesh Loading & Rendering:**
    * Defines a mesh as a collection of triangles and loads in a pre-defined mesh (Pikachu!).
    * You'll implement a function to raytrace the mesh, finding the closest intersecting triangle for each ray and rendering the final image.
* **Bonus: Testing with `pytest`, More Raytracing:**
    * Introduces the `pytest` library for writing and running tests, showcasing its benefits for modularizing and parameterizing tests.
    * Suggests additional exercises to further explore raytracing, such as rendering videos, optimizing on the GPU, adding color, and using multiple rays per pixel for smoother edges.

**Section 3: CNNs & ResNets - Building Blocks for Image Recognition**

This section focuses on understanding and implementing two important neural network architectures: Convolutional Neural Networks (CNNs) and Residual Networks (ResNets).

* **Making Your Own Modules:**
    * Reinforces the concept of subclassing `nn.Module` to create custom PyTorch modules.
    * Guides you through implementing basic modules like `ReLU` and `Linear` layers, including initialization, forward pass, and the `extra_repr` method for informative string representations.
    * Introduces the `nn.Parameter` class for storing learnable weights and biases.
    * You'll assemble these modules into a simple Multi-Layer Perceptron (MLP).
* **Training:**
    * Covers the process of training your MLP on the MNIST dataset, introducing concepts like:
        * Transforms, datasets, and dataloaders for efficient data loading and preprocessing.
        * The structure of a training loop, including forward pass, loss calculation, backpropagation, optimizer step, and gradient zeroing.
        * Writing a validation loop to evaluate model performance on unseen data.
    * Introduces the use of `dataclass` to organize training arguments.
* **Convolutions:**
    * Explains the concept of convolutions, their application in vision models, and benefits like parameter sharing and locality exploitation.
    * You'll implement `Conv2d` and `MaxPool2d` modules using PyTorch's functional versions (`t.nn.functional.conv2d` and `t.nn.functional.max_pool2d`) and understand their functionalities.
* **ResNets:**
    * Introduces the ResNet architecture and its key innovation - **skip connections**, which help alleviate the vanishing gradient problem and enable training of very deep networks.
    * Covers batch normalization and its role in improving training stability.
    * You'll assemble your own ResNet34 model, load pre-trained weights from PyTorch's implementation, and use it to classify images.
* **Bonus - Convolutions From Scratch, Feature Extraction:**
    * Offers optional exercises to implement convolutions and maxpooling from scratch using `as_strided` and stride-based methods, providing a deeper understanding of how these operations work.
    * Explores **feature extraction** - repurposing a pre-trained ResNet for a new task by freezing most layers and training only the final layers.

**Section 4: Optimization & Hyperparameters - Fine-tuning Model Training**

This section focuses on understanding and implementing various optimization algorithms and techniques for improving model training and hyperparameter selection.

* **Optimizers:**
    * Introduces popular optimization algorithms used in deep learning, including:
        * **Stochastic Gradient Descent (SGD):**  The classic optimization algorithm, which iteratively updates parameters based on their gradients.
        * **RMSprop:**  An algorithm that adapts the learning rate for each parameter based on the magnitude of its recent gradients.
        * **Adam:**  A widely used algorithm that combines momentum and RMSprop, offering fast convergence and good generalization.
    * You'll implement these algorithms from scratch and visualize their performance on various loss landscapes (including "pathological" ones that illustrate challenges in optimization).
    * Explores concepts like **momentum**, **weight decay**, and the role of hyperparameters in controlling the behavior of these algorithms.
* **Weights and Biases:**
    * Introduces Weights and Biases (WandB) - a powerful tool for tracking experiments, visualizing training progress, and performing hyperparameter sweeps.
    * You'll adapt the training loop from the previous section to log training metrics and model outputs to WandB, enabling efficient comparison of different runs and hyperparameter configurations.
    * Covers using WandB's sweep functionality to automatically test different hyperparameter combinations and identify the best performing ones.
* **Bonus:**
    * Suggests further exploration of concepts like **scaling laws** (how model performance changes with dataset size and model size), 
    * Encourages experimentation with additional WandB features (logging media, saving/loading models), training models from scratch, and delving into the concept of the **Optimizer's Curse** (the tendency for hyperparameter tuning to overfit to the validation set).

**Section 5: Backpropagation - Demystifying Gradient Calculation**

This section dives into the heart of deep learning - **backpropagation**, the algorithm that enables efficient calculation of gradients in complex neural networks.

* **Introduction:**
    * Reviews the backpropagation algorithm and its role in training deep learning models.
    * Explains the concept of a **computational graph** - a representation of the operations involved in computing the model's output, used to efficiently track and calculate gradients.
    * Introduces the idea of **backward functions** - functions that calculate the gradient of a specific operation with respect to its inputs, used to backpropagate gradients through the graph.
* **Autograd:**
    * Dives deeper into the backpropagation algorithm, explaining the importance of **topological sorting** - ordering the nodes in the graph to ensure that gradients are calculated in the correct order.
    * You'll implement the `backprop` function, which iteratively calculates and stores gradients for all tensors in the computational graph, mimicking PyTorch's `backward()` method.
* **More Forward & Backward Functions:**
    * Extends the set of functions supported by your backpropagation system, implementing the backward functions for operations like:
        * `negative`, `exp`, `reshape`, `permute`, `expand`, `sum`, `indexing`, and elementwise operations (add, subtract, divide).
    * Introduces the concept of **in-place operations** and the challenges they pose for backpropagation (modifying inputs can lead to incorrect gradient calculations).
    * You'll implement a warning system to handle these cases.
* **Putting Everything Together:**
    * Completes the process of building your own deep learning framework by implementing equivalents of PyTorch's `nn.Parameter` and `nn.Module` classes.
    * You'll build your own linear layer (`Linear`) using these classes and your backpropagation system.
    * Implements cross-entropy loss and a `NoGrad` context manager (similar to `torch.no_grad`) to disable gradient tracking during inference.
    * Finally, you'll combine all the pieces to train your own MLP on the MNIST dataset using your custom backpropagation framework.
* **Bonus:**
    * Suggests further extensions to your framework, such as:
        * Implementing more complex operations (e.g., `einsum`).
        * Supporting differentiation with respect to keyword arguments.
        * Adding central difference checking to verify gradient calculations.

**Section 6: GANs & VAEs - Exploring Generative Models**

This section explores two powerful generative models: Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), showcasing their ability to generate new data resembling the training data.

* **Introduction:**
    * Introduces the concepts of GANs and VAEs, their differences, and common applications in generating images and other data.
    * Discusses transposed convolutions (used for upsampling in generative models) and the reparameterization trick for VAEs.
* **GANs:**
    * Dives deeper into GANs, explaining the adversarial training process between the generator and discriminator networks.
    * Covers the DCGAN architecture and its application in generating celebrity images.
    * You'll implement and train your own DCGAN, exploring techniques for improving convergence and generating realistic images.
    * Discusses challenges in training GANs, such as mode collapse and instability.
* **Autoencoders & VAEs:**
    * Introduces Autoencoders - networks that learn to compress data into a lower-dimensional representation (encoding) and then reconstruct it (decoding).
    * Explains how VAEs extend autoencoders by imposing a probabilistic structure on the latent space, enabling them to generate new, diverse samples.
    * You'll implement and train your own autoencoder and VAE, exploring their capabilities in compressing and generating MNIST images.
    * Discusses the **reparameterization trick**, which enables backpropagation through the sampling process in VAEs.
* **Bonus - Transposed Convolutions:**
    * Offers optional exercises to implement transposed convolutions from scratch using stride-based methods and low-level tensor manipulation, mirroring the bonus section for convolutions in the "CNNs & ResNets" section.
    * Challenges you to build GANs and VAEs entirely from scratch, using your custom modules.

Throughout this chapter, you'll transition from building simple modules to assembling complex architectures, training them on various datasets, and analyzing their inner workings. By the end, you'll have a strong foundational understanding of deep learning and the skills to dive deeper into the specialized topics covered in the later chapters of the ARENA 3.0 program.
