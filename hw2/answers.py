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

Yes, the graphs match what we expected to see.

**Explanation:**
- **Without dropout (blue line)**: The model achieves ~95% training accuracy while test accuracy remains stuck at ~23%. This is classic **severe overfitting**. The model memorizes the training set perfectly but completely fails to generalize to unseen test data. The train loss decreases to ~0.3 while test loss increases over time, diverging dramatically.

- **With dropout (orange and green lines)**: Adding dropout constrains the model from overfitting. With dropout=0.4, training accuracy is reduced to ~70%, but test accuracy improves significantly to ~29% - a 26% improvement over no dropout! This trade-off is exactly what we expect: dropout prevents co-adaptation of neurons and forces the network to learn more robust features.

**Examples from graphs:**
- Train loss (log scale): No dropout drops to ~0.3; dropout=0.4 stays around ~1.0
- Test loss (log scale): No dropout increases to ~2.8; dropout=0.4 stays stable around ~2.1
- Train-test gap: No dropout has 95%-23%=72% gap; dropout=0.4 has only 70%-29%=41% gap

## Part 2: Low-Dropout vs High-Dropout Comparison

**Dropout=0.4 (orange) vs Dropout=0.8 (green):**

- **Dropout=0.4**: Best performance with ~70% train accuracy and ~29% test accuracy. Training curves are smooth and test loss is stable. This is the **sweet spot** for regularization.

- **Dropout=0.8**: Underfitting occurs. Train accuracy only reaches ~30% because dropping 80% of neurons is too aggressive - the network loses too much capacity to learn. Test accuracy (~22-24%) is worse than dropout=0.4 because the model fundamentally can't learn the patterns in the data.

**Key insight**: There's an optimal dropout rate. Too little (0) causes overfitting, too much (0.8) causes underfitting. Dropout=0.4 balances regularization and capacity, providing the best generalization in this setting.
"""

part2_q2 = r"""
**Your answer:**

Yes, it is absolutely possible for test loss to **decrease** while test accuracy **decreases** simultaneously.

## Why This Happens

This can occur due to **changes in model confidence and probability distribution**, not necessarily better predictions.

### Example Scenario:

Consider a classification where:
- **Epoch 1**: Model predicts class 0 with confidence 60% and class 1 with confidence 40% for a sample that is actually class 1
  - Loss: $-\log(0.4) = 0.916$ (incorrect but somewhat confident)
  - Accuracy: 0 (wrong prediction)

- **Epoch 2**: Same model predicts class 0 with confidence 55% and class 1 with confidence 45% 
  - Loss: $-\log(0.45) = 0.799$ (loss decreased!)
  - Accuracy: 0 (still wrong prediction)

### Root Cause:

The cross-entropy loss measures the quality of probability estimates, **not just whether predictions are correct**. When the model:
1. Reduces confidence in its (incorrect) top prediction
2. Increases confidence in the correct class slightly
3. But the predicted class remains the same

Then loss can decrease while accuracy stays the same or even decreases if the model becomes less confident in correct predictions overall.

This is particularly common when:
- The model is overfitting and learning to be overconfident
- Regularization or dropout starts to reduce overconfidence
- The model redistributes its probability mass in ways that improve calibration but hurt accuracy

In practice, when validation loss decreases but validation accuracy stagnates or decreases, it often indicates the model is improving its **confidence calibration** rather than its **classification ability**, which can be a sign of good regularization taking effect.
"""

part2_q3 = r"""
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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
    raise NotImplementedError()
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