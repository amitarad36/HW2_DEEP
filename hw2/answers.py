r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1. The Jacobian has one entry for every element of **Y** with respect to every element of **X**.
   Since `Y.shape = (64,512)` and `X.shape = (64,1024)`, the full Jacobian tensor is:

   $$
   \frac{\partial \mathbf{Y}}{\partial \mathbf{X}} \in 
   \mathbb{R}^{64 \times 512 \times 64 \times 1024}
   $$

2. Viewing the Jacobian as a **block matrix** of size $64 \times 64$, where each block is 
   $(\text{out\_features} \times \text{in\_features}) = (512 \times 1024)$, we observe:

   * Off-diagonal blocks are **zero matrices**, because sample *n* depends only on sample *n*.
   * Each diagonal block equals the weight matrix $W$.

   Therefore the Jacobian is **block diagonal**, with identical diagonal blocks.

3. Instead of representing the full tensor, we may keep only the non-zero blocks.
   This yields a reduced tensor:

   $$
   (N,\,\text{out},\,\text{in}) = (64, 512, 1024)
   $$

   And because each block is exactly $W$, the most compressed representation is just:

   $$
   \boxed{W \in \mathbb{R}^{512 \times 1024}}
   $$

   plus the knowledge that the Jacobian is $I_{64} \otimes W$.
   
4. **Computing** $\delta \mathbf{X}$ **from** $\delta \mathbf{Y}$ **(without the Jacobian)**

   For the linear layer:
   $$
   \mathbf{Y} = \mathbf{X} \mathbf{W}^\top + \mathbf{b}
   $$

   and upstream gradient:
   $$
   \delta \mathbf{Y} = \frac{\partial L}{\partial \mathbf{Y}},
   $$

   each sample satisfies:
   $$
   \mathbf{y}_n = W \mathbf{x}_n^\top + \mathbf{b}
   \quad\Rightarrow\quad
   \frac{\partial \mathbf{y}_n}{\partial \mathbf{x}_n} = W
   $$

   so the chain rule gives:
   $$
   \delta \mathbf{x}_n = \delta \mathbf{y}_n\, W
   $$

   Stacked across the batch:
   $$
   \boxed{\delta \mathbf{X} = \delta \mathbf{Y} \, W}
   $$

   which corresponds exactly to the implementation:
   ```python
   dx = dout @ W
   ```

5. **Jacobian of** $\mathbf{Y}$ **w.r.t. the weights** $\mathbf{W}$

   We have:
   $$
   \mathbf{Y} = \mathbf{X}\mathbf{W}^\top + \mathbf{b}, \qquad
   \mathbf{Y}\in\mathbb{R}^{64\times512}, \quad
   \mathbf{W}\in\mathbb{R}^{512\times1024}.
   $$

   Therefore the full Jacobian tensor is:
   $$
   \frac{\partial \mathbf{Y}}{\partial \mathbf{W}}
   \in \mathbb{R}^{64 \times 512 \times 512 \times 1024},
   $$
   
   where the indices correspond to:
   $$
   (n,\; o,\; o',\; i) 
   \quad\longrightarrow\quad
   \frac{\partial Y_{n,o}}{\partial W_{o',i}}.
   $$

   ### Block-matrix interpretation

   If we flatten by grouping rows by output index $o$ and columns by weight-row $o'$,  
   the Jacobian may be viewed as a:
   $$
   \boxed{512 \times 512 \text{ block matrix}},
   $$
   
   where each block has shape:
   $$
   \boxed{64 \times 1024} \qquad
   (\text{batch size} \times \text{weight columns}).
   $$

   Each element is:
   $$
   \frac{\partial Y_{n,o}}{\partial W_{o',i}}
   =
   \begin{cases}
   X_{n,i}, & o = o' \\
   0, & o \ne o'
   \end{cases}
   $$

   Hence:

   * Only the diagonal blocks ($o = o'$) are non-zero.  
   * Each non-zero block contains input rows $X_{n,:}$.  
   * The Jacobian is block-diagonal with $512$ meaningful diagonal blocks,  
     each of size $64 \times 1024$.
"""

part1_q2 = r"""
**Your answer:**

Gradient descent only looks at the gradient, so it looks at the loss surface uniformly in every direction. It might become inefficient when the function has very different curvature in different directions — for example, a long narrow valley where one direction is very steep and the other is almost flat.  
In this situation GD keeps jumping from side to side across the steep direction and moves very slowly along the flat direction. So the optimizer spends most of its time “zig-zagging” instead of actually making progress.

This is exactly where the second-order derivative (the Hessian) becomes useful.  
The Hessian tells us how curved the function is in each direction:

- **High curvature** → the loss changes quickly (steep direction).  
- **Low curvature** → the loss changes slowly (flat direction).

Knowing this lets us adjust the step size differently for different directions: take tiny steps where the curvature is high, and much larger steps where the curvature is low.

Newton's method is an example of using this idea. The update rule

$$
\theta_{t+1} = \theta_t - H^{-1} \nabla J(\theta_t)
$$

basically says: "scale the gradient by the inverse curvature."  (separately for each direction)
This helps the optimizer take more direct paths to the minimum, avoiding the zig-zagging problem.

"""




# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1 
    lr = 0.05
    reg = 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    
    lr_vanilla = 0.02      
    lr_momentum = 0.005     
    lr_rmsprop = 0.0005    
    
    reg = 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1  
    lr = 0.001 
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

## Part 1: No-Dropout vs Dropout

Yes, the graphs match expectations.

**What happens:**
- **No dropout**: The model memorizes the small train set, so train accuracy goes up while test accuracy goes down. Train loss keeps falling but test loss is going upward, displaying bad generalization.

- **With dropout**: Dropping units forces the model to spread the learning across more neurons. Training improves more slowly, but the test curve stays better, test accuracy is higher than with no dropout, and the train–test gap shrinks.

## Part 2: Low-Dropout vs High-Dropout Comparison

- **Moderate dropout**: Good middle ground—still learns useful patterns but doesn’t memorize, so train and test stay closer.
- **High dropout**: Drops too many neurons that learning slows down, train accuracy doesn't improve, and test accuracy doesn't improve much either.

**conclusion:** No dropout overfits, too much dropout underfits. A medium setting is the one to look for.
"""

part2_q2 = r"""
**Your answer:**

Yes, this is possible.

## How it might happen

Accuracy is just right or wrong, but cross-entropy loss measures how confident the model is with its predictions. The model can get really confident on samples it's already classifying correctly, which drops the loss a lot, while at the same time nudging a few on-the-margin examples from correct to incorrect, which drops the accuracy.

## Example

Say we have 2 test samples:

- **Epoch 1**: Model predicts both correctly but with low confidence (like 51% on the right class).  
  - Accuracy: 100%  
  - Loss: High (cross-entropy: $-\log(0.51) + (-\log(0.51)) \approx 0.67 + 0.67 = 1.34$)

- **Epoch 2**: Model becomes 99% confident on the first sample (correct), but drops to 49% on the second sample (now wrong).  
  - Accuracy: 50% (dropped)  
  - Loss: Lower (cross-entropy: $-\log(0.99) + (-\log(0.49)) \approx 0.01 + 0.71 = 0.72$)

So loss dropped from 1.34 to 0.72, even though accuracy dropped from 100% to 50%. This happens when the model makes a new mistake but gets way more confident on other samples.
"""

part2_q3 = r"""
**Your answer:**

## 1. GD vs SGD

**Gradient Descent (GD):**
- Uses the **entire dataset** to compute the gradient at each single step
- Update rule: $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta; \mathcal{D})$ where $\mathcal{D}$ is all $N$ training samples
- The gradient is exact,  so it points directly toward the true (local) minimum
- Very slow when the dataset is large (needs to compute loss and gradient over all samples)
- Each step is more "certain" but it takes fewer of them per unit time

**Stochastic Gradient Descent (SGD):**
- Uses a **mini-batch** which is a chunk of the dataset to estimate the gradient of the loss at each step
- Update rule: $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta; \mathcal{B})$ where $\mathcal{B}$ is a small batch
- The gradient varies from batch to batch (hypothetically because we calculate it in a different location)
- Much faster per epoch (small batches fit in memory, compute quickly)
- Takes many more inaccurate steps, but can escape shallow local minima due to the noise

**Similarities:**
- Both follow the negative gradient direction on average
- Both use a learning rate $\eta$ to control step size
- Both converge to a local minimum (under proper conditions)
- The core idea is identical: move downhill on the loss surface

**Key difference:** 
- GD is deterministic and smooth whdile SGD is stochastic and noisy
- SGD gives up precision per step for many fast steps while GD takes fewer but more accurate steps

## 2. using momentum with GD?

**Yes, momentum can help GD, but it's not as critical as it is for SGD.**

**Why momentum helps GD:**
- GD still deals with valleys, flat spots, and saddle points even though the gradient is exact
- Momentum keeps things moving when the gradient gets tiny or keeps changing direction a bit
- It helps the optimizer "build up speed" and push through flat areas instead of crawling

**Why it's less important than for SGD:**
- SGD's probem is that gradients are noisy since computing different batches gives other gradients - momentum can help overcome that noise and keeps the updates moving in the general right direction
- GD already has smooth, exact gradients, so momentum is less crucial
- The main win for GD is avoiding getting stuck in local minima created by weird loss landscapes; for SGD it's dealing with the noise from sampling batches

**conclusion:** Momentum helps both, but SGD really improves when dealing with noisy batches. For GD it's helpful for handling irregular shapes in the loss surface.

## 3. Handling memory constraints with GD

### 3.1 Gradient equivalence

**Yes, this approach yeilds a gradient equivalent to GD.**

Mathematicaly relying on the linearity of differentiation.

**GD Gradient:** Calculates the gradient of the total loss, which is the sum of losses over all samples:
$$\nabla_\theta L_{\text{total}} = \nabla_\theta \left( \sum_{i=1}^{N} L(x_i) \right)$$

**New approach:** Sums the losses of the disjoint batches first, then takes the gradient:
$$\nabla_\theta (L_{\text{batch1}} + L_{\text{batch2}} + \dots) = \nabla_\theta L_{\text{batch1}} + \nabla_\theta L_{\text{batch2}} + \dots$$

Since the batches are just disjoint groups of the original samples, the sum of the batch gradients is equal to the gradient of the sum.

### 3.2 Out of memory error

The problem with this method ensues from the Computational Graph.

When performing a forward pass, all intermediate values are stored in memory (activations, intermediate results) so they are available during backpropagation.
By waiting to call `backward()` until after all batches, one forces the memory to keep the entire "history" for every single batch at the same time.it's just fitting the entire dataset's computation graph into RAM, but accumulated one batch at a time.

### 3.3 How to solve it: Gradient Accumulation

We can try accumulating gradients instead of accumulating the *loss*. Doing so allows us to free the computation graph after each batch, since we only hold the gradients per batch in memory.

**The Corrected Procedure:**
for each batch:
   - Forward pass it
   - Calculate loss
   - Immediately call backward()
   - Accumulate gradients

The parameter gradients now contain the sum of gradients which is equivalent to gradient of the total loss.
Call `optimizer.step()` to update parameters
Call `optimizer.zero_grad()` to reset gradients

"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 8
    hidden_dims = 15
    activation = "relu"
    out_activation = "none"


    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======

    loss_fn = torch.nn.CrossEntropyLoss()
    lr, weight_decay, momentum = 0.1, 0, 0.05

    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
### 1. Types of error

**1.1 optimization error:** ensues from the imperfections of the optimization algorithm and the incomplete picture the data provides (sometimes reffered to as Estimation Error).

**1.2 generalization error:** is the expected error on unseen data, for example, hen a model overfits, it has learned the training data too well, including its noise and outliers, and thus performs poorly on new data.

**1.3 approximation error:** is the error that arises from the limited expresivity of the model, for example, trying to fit a linear approximator to a non-linear data relationship.


### 2. Errors on the plots
**2.1 optimization error:**
"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.01
    weight_decay = 0.001
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""