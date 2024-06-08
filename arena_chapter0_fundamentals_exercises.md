## Deep Dive into Chapter 0: Fundamentals Exercises -  Building a Strong Deep Learning Foundation 

The `chapter0_fundamentals/exercises` folder in the ARENA 3.0 repository is a treasure trove of hands-on coding exercises designed to solidify your understanding of core deep learning concepts and equip you with the practical skills to build, train, and analyze neural networks. 

Here's a detailed breakdown of the exercises within this chapter:

**Part 0: Prerequisites (`part0_prereqs`)**

This section serves as a warm-up, ensuring you have a firm grasp of basic tensor manipulation and linear algebra operations using essential libraries like `einops` and `einsum`. 

* **File:** `numbers.npy`
    * Contains a NumPy array with a specific shape, used as input for the following exercises.
* **File:** `solutions.py`
    * Provides solutions to the exercises.
* **File:** `tests.py`
    * Contains test functions to verify your implementations.
* **File:** `utils.py`
    * Provides helper functions for visualizing and manipulating the `numbers.npy` array.

**Key Exercises:**

1. **Einops & Einsum:**  
    * You'll perform various tensor transformations on the `numbers.npy` array using `einops` functions like `rearrange`, `repeat`, and `reduce` to reshape, tile, and aggregate elements.
    * You'll also implement common linear algebra operations (trace, matrix-vector multiplication, matrix multiplication, inner/outer product) using `einops.einsum`, showcasing its flexibility and conciseness.

**Part 1: Ray Tracing (`part1_ray_tracing`)**

This section uses a basic graphics rendering task (ray tracing) to introduce you to key concepts in linear algebra, PyTorch tensor manipulation, and the power of batched operations. 

* **Files:** 
    * `pikachu.pt`: Contains a PyTorch tensor representing a 3D mesh of Pikachu, which you'll render later.
    * `pikachu.stl`: An STL file of the Pikachu mesh (can be visualized with 3D modeling software). 
    * `solutions.py`: Solutions to the exercises.
    * `tests.py` and `test_with_pytest.py`: Contains test functions to verify your implementations.
    * `utils.py`:  Provides helper functions for visualizing rays and triangles.

**Key Exercises:**

1. **1D Image Rendering:** 
    * You'll implement a function `make_rays_1d` to generate rays in 2D space, introducing the concepts of a camera, screen, rays, and their representation as origin and direction points.
2. **Ray-Object Intersection:** 
    * You'll implement `intersect_ray_1d` to determine if a given ray intersects a line segment (representing a simplified object in the scene). 
    * This exercise requires solving a linear system of equations and checking if the solution lies within the bounds of the ray and the line segment.
3. **Batched Ray-Segment Intersection:** 
    * You'll implement a more efficient version of the previous exercise, `intersect_rays_1d`, which handles multiple rays and segments simultaneously using batched operations in PyTorch. This introduces concepts like broadcasting, logical reductions, and the `einops` library for efficient tensor manipulation.
4. **2D Rays:** 
    * You'll extend your ray generation function to create `make_rays_2d`, which generates 3D rays emanating from the camera and passing through a 2D screen. 
    * This lays the foundation for rendering a 3D mesh onto a 2D image.
5. **Triangle Coordinates & Intersection:** 
    * You'll learn about parameterizing triangles using **barycentric coordinates** and implement `triangle_ray_intersects` to determine if a ray intersects a triangle in 3D space.
    * This requires solving a linear system of equations and checking if the solution lies within the bounds of the triangle.
6. **Single-Triangle Rendering:** 
    * You'll implement `raytrace_triangle`, which takes a batch of 3D rays and a single triangle and returns a boolean tensor indicating whether each ray intersects the triangle.
    * This function combines the concepts from the previous exercises, and you'll visualize the result as a 2D image.
7. **Mesh Rendering:** 
    * This exercise is the culmination of the previous ones. You'll implement `raytrace_mesh`, which takes a batch of 3D rays and a 3D mesh (represented as a collection of triangles) and returns a tensor containing the distance to the closest intersecting triangle for each ray.
    * You'll then use this function to render your own Pikachu image from the provided mesh file, showcasing the power of ray tracing and batched operations.

**Part 2: CNNs & ResNets (`part2_cnns`)**

This section focuses on building and training convolutional neural networks (CNNs), and specifically Residual Networks (ResNets), a powerful architecture for image classification. 

* **Files:**
    * `imagenet_labels.json`:  Contains a mapping of ImageNet class labels to their descriptions.
    * `resnet_inputs/`:  Contains sample images to test your ResNet with.
    * `solutions.py` and `solutions_bonus.py`:  Solutions to the exercises.
    * `tests.py`:  Contains test functions to verify your implementations.
    * `utils.py`:  Provides helper functions for analyzing and visualizing model parameters.

**Key Exercises:**

1. **Making your own modules:** 
    * You'll solidify your understanding of `nn.Module` by implementing several basic building blocks of neural networks:
        * `ReLU`:  A simple activation function.
        * `Linear`: A linear layer with weights and optional bias.
        * `Flatten`: A module to reshape tensors for feeding into linear layers.
    * This exercise emphasizes proper initialization of weights and biases, understanding forward pass logic, and using `extra_repr` for informative string representations.
    * You'll assemble these modules into a simple MLP.
2. **Training:** 
    * You'll train your MLP on the MNIST dataset, learning how to:
        * Use `torchvision.datasets` and `torch.utils.data.DataLoader` for efficient data loading and batching.
        * Implement a basic training loop with gradient descent, using an optimizer (`t.optim.Adam`) to update model parameters.
        * Write a validation loop to evaluate performance on a separate test set.
    * This exercise introduces the concept of a **training epoch**, measuring loss during training, and computing accuracy on the test set.
3. **Convolutions:** 
    * You'll delve into the mechanics of convolutions, learning about:
        * The convolutional operation itself and its benefits for image recognition (parameter sharing, locality).
        * Implementing `Conv2d` and `MaxPool2d` modules, either using PyTorch's built-in functional versions or from scratch (optional bonus exercise).
4. **ResNets:** 
    * This exercise combines all the pieces you've learned so far to build a **ResNet34** model. 
    * You'll understand the significance of **skip connections** in overcoming the vanishing gradient problem and enabling training of very deep networks.
    * You'll implement **Residual Blocks** and **Block Groups** to create the ResNet architecture, learning about techniques like **batch normalization** for improving training stability.
    * You'll then load pre-trained weights from PyTorch's ResNet34 implementation and use your model to classify images from the ImageNet dataset.
5. **Bonus - Convolutions From Scratch, Feature Extraction:** 
    * **Convolutions From Scratch:** This optional section challenges you to implement `Conv2d` and `MaxPool2d` entirely from scratch, using low-level tensor manipulations and array strides. This will solidify your understanding of how these operations work at a fundamental level.
    * **Feature Extraction:**  This bonus section explores repurposing your ResNet for a different task (classifying CIFAR-10 images) using **feature extraction**, where you freeze most layers and train only the final linear layer on the new dataset.

**Part 3: Optimization (`part3_optimization`)**

This section delves deeper into optimization algorithms and their role in training deep learning models, focusing on fine-tuning model training and efficiently searching for optimal hyperparameters.

* **Files:**
    * `solutions.py` and `solutions_bonus.py`:  Solutions to the exercises.
    * `tests.py`:  Contains test functions to verify your implementations.
    * `utils.py`:  Provides helper functions for visualizing loss landscapes and plotting results.

**Key Exercises:**

1. **Optimizers:**
    * You'll gain a deeper understanding of three popular optimization algorithms:
        * **SGD:**  Implement SGD with momentum and weight decay.
        * **RMSprop:** Implement RMSprop with momentum and weight decay.
        * **Adam:**  Implement Adam with weight decay.
    * You'll visualize how these optimizers traverse different loss landscapes, observing the effects of momentum, learning rates, and weight decay on their performance.
    * You'll also experiment with training a model on a "pathological" loss landscape (one with challenging geometry), comparing the effectiveness of different optimizers.
2. **Weights & Biases:**
    * This section introduces you to Weights & Biases (WandB), a powerful tool for tracking experiments, visualizing results, and performing hyperparameter sweeps.
    * You'll adapt the ResNet training loop from the previous section to log metrics and results to WandB, enabling easy comparison of different runs and hyperparameter configurations.
    * You'll learn how to define a **sweep configuration** in WandB, specifying the range and distribution of hyperparameters to test, and run sweeps to identify the best-performing configurations.

**Part 4: Backpropagation (`part4_backprop`)**

This section dives into the heart of deep learning, challenging you to build your own backpropagation system from scratch, mirroring PyTorch's `autograd` functionality.

* **Files:**
    * `solutions.py`:  Solutions to the exercises.
    * `tests.py`:  Contains test functions to verify your implementations.
    * `utils.py`:  Provides helper functions for visualizing data and loading the MNIST dataset.

**Key Exercises:**

1. **Introduction to backprop:**
    * You'll learn about **computational graphs**, the fundamental data structure used to represent operations in a neural network and track dependencies for gradient calculation.
    * You'll implement the backward function for `log`, demonstrating how to calculate the gradient of an operation with respect to its inputs.
2. **Autograd:**
    * You'll delve deeper into the backpropagation algorithm, understanding the importance of **topological sorting** to calculate gradients in the correct order.
    * You'll implement a `backprop` function, which calculates and stores gradients for all tensors in a computational graph, mirroring PyTorch's `backward()` method.
3. **More forward & backward functions:**
    * You'll expand your backprop system by implementing the forward and backward functions for various operations:
        * `negative`, `exp`, `reshape`, `permute`, `expand`, `sum` (with different arguments), `indexing`, `add`, `subtract`, `true_divide`, `maximum`, and `relu`.
    * You'll learn how to handle broadcasting and implement the `unbroadcast` function to reverse broadcasting during backpropagation.
    * You'll also handle the challenge of **in-place operations**, where modifying inputs can lead to incorrect gradient calculations, and implement warning messages for these cases.
4. **Putting everything together:**
    * You'll implement your own `Parameter` and `Module` classes, mirroring PyTorch's corresponding classes, enabling you to define and organize parameters within your network.
    * You'll then build a `Linear` layer using these classes and your custom backpropagation system.
    * You'll also implement `cross_entropy` loss and a `NoGrad` context manager to disable gradient tracking during inference.
    * Finally, you'll train a simple MLP on the MNIST dataset using your own hand-built backpropagation framework, bringing the entire section full circle.
5. **Bonus:**
    * This section suggests additional exercises for extending your framework:
        * Implementing support for the `einsum` operation.
        * Handling differentiation with respect to keyword arguments.
        * Replicating PyTorch's behavior for reusing modules during the forward pass.
        * Implementing convolutional layers using the foundations you've built.
        * Adding support for ResNet architectures.
        * Using central difference checking to verify your gradient calculations against a numerical approximation.
        * Adding support for non-differentiable functions.
        * Implementing `torch.stack` and its backward function.

**Part 5: GANs & VAEs (`part5_gans_and_vaes`)**

This section introduces two powerful generative models: Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), and challenges you to implement and train them. 

* **Files:**
    * `solutions.py`: Solutions to the exercises.
    * `tests.py`: Contains test functions to verify your implementations.

**Key Exercises:**

1. **GANs:**
    * You'll learn about the adversarial training process in GANs, where a generator network learns to produce realistic data and a discriminator network learns to distinguish between real and generated data.
    * You'll implement and train a **DCGAN** (Deep Convolutional GAN) to generate images of celebrities. 
    * The exercises cover building the generator and discriminator networks, understanding the loss functions and training process, and visualizing the generated images.
    * You'll also learn about common challenges in GAN training, like **mode collapse**, and explore techniques for mitigating these issues.
2. **Autoencoders & VAEs:**
    * This exercise explores **autoencoders** - networks that learn to compress data into a lower-dimensional latent space and then reconstruct it. 
    * You'll implement a basic autoencoder, understanding its encoder and decoder components, and train it on the MNIST dataset.
    * You'll then extend this concept to **VAEs**, learning about how they impose a probabilistic structure on the latent space, enabling them to generate new and diverse samples.
    * You'll implement a VAE, training it on MNIST and visualizing the generated images.
    * This exercise introduces the **reparameterization trick** for VAEs, which enables backpropagation through the sampling process.
3. **Bonus - Transposed Convolutions:**
    * This optional section focuses on understanding and implementing **transposed convolutions**, which are used in upsampling operations in generative models (e.g., going from a low-dimensional latent vector to a high-resolution image).
    * You'll implement the transposed convolution operation from scratch, building upon the knowledge you gained from implementing convolutions in the previous sections.
    * Finally, you'll be challenged to implement GANs and VAEs entirely from scratch, using your custom transposed convolution modules.

This final section of Chapter 0 introduces the fascinating world of generative models, showcasing the power of deep learning to create new data and pushing your understanding of neural networks further.


Overall, the `chapter0_fundamentals/exercises` content provides a comprehensive and challenging journey through the essential building blocks of deep learning. These exercises will build your confidence working with tensors, implementing and training various neural network architectures, understanding backpropagation, and even delving into generative models. By the end of this chapter, you'll have a strong foundation to tackle the more specialized and advanced topics of transformer interpretability and reinforcement learning.
