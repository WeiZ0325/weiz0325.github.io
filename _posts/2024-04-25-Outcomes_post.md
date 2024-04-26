# Outcomes from fastai course

1. TOC
{:toc}

## Deployment
This section states the process of deployment for fastai course includes resources and links

### Resources
1. Notebook - saving a basic fastai model:
- [Kaggle](https://www.kaggle.com/code/jhoward/saving-a-basic-fastai-model)
- [Colab](https://colab.research.google.com/drive/1M-mzhZdFQ2XWBSbLCuKzrmLsm0aLEYxQ?usp=sharing)
2. [Chapter 2 notebook](https://github.com/fastai/fastbook/blob/master/02_production.ipynb)
3. [Solutions to chapter 2 questions from the book](https://forums.fast.ai/t/fastbook-chapter-2-questionnaire-solutions-wiki/66392)

### Links
1. [Gradio tutorial](https://www.tanishq.ai/blog/posts/2021-11-16-gradio-huggingface.html) from @ilovescience
2. [HF Spaces](https://huggingface.co/spaces)
3. Installing a python environment
- [fastsetup](https://github.com/fastai/fastsetup)
- Windos: [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and [Terminal](https://apps.microsoft.com/detail/9n0dx20hk701?hl=en-gb&gl=AU)
- tinypets [github](https://github.com/fastai/tinypets) / [site](https://fastai.github.io/tinypets/)
- tinypets fork [github](https://github.com/jph00/tinypets) / [site](https://jph00.github.io/tinypets/)

## Neural net foundations
[Chapter 4](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb)

**Reflecting** on the ["*Under the Hood: Training a Digit Classifier*"](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb) chapter, it becomes evident that the foundational elements of deep learning are intertwined, much like the stones in an arch, each one crucial for the stability and strength of the overall structure. This metaphor not only underscores the interdependency of core concepts like tensors, SGD, and loss functions but also captures the essence of the learning process—complex yet formidable when all pieces align.

The historical anecdotes serve as a powerful reminder that the field of AI, as with any pioneering endeavor, is paved with the perseverance of those who dare to continue despite obstacles and skepticism. The stories of Lecun, Hinton, Bengio, Schmidhuber, and Werbos are not just narratives of academic persistence; they are testament to the unyielding human spirit that drives innovation. They exemplify that breakthroughs often lie just beyond the fringe of collective belief and that the fruits of tenacity can be far-reaching, even when recognition and success are delayed.

This reflection leads to an appreciation of patience and resilience, both in understanding deep learning's complexities and in contributing to its future. It highlights a truth relevant to learners and experts alike: mastery of deep learning is not merely a technical endeavor but a journey that may require the courage to forge ahead when faced with the unknown or the unaccepted. The chapter instills a sense of hope and determination—to keep learning, experimenting, and pushing boundaries, as this is the path tread by those who have left indelible marks on the landscape of artificial intelligence.

1. How is greyscale image represented on a computer? How about a color image?

Images are represented by arrays with pixel values representing the content of the image. For grayscale images, a 2-dimensional array is used with the pixels representing the grayscale values, with a range of 256 integers. A value of 0 represents black, and a value of 255 represents white, with different shades of gray in between. For color images, three color channels (red, green, blue) are typically used, with a separate 256-range 2D array used for each channel. A pixel value of 0 represents black, with 255 representing solid red, green, or blue. The three 2D arrays form a final 3D array (rank 3 tensor) representing the color image.

2. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?


There are two subfolders, `train` and `valid`, the former contains the data for model training, the latter contains the data for validating model performance after each training step. Evaluating the model on the validation set serves two purposes: a) to report a human-interpretable metric such as accuracy (in contrast to the often abstract loss functions used for training), b) to facilitate the detection of overfitting by evaluating the model on a dataset it hasn’t been trained on (in short, an overfitting model performs increasingly well on the training set but decreasingly so on the validation set). Of course, every practicioner could generate their own train/validation-split of the data. Public datasets are usually pre-split to simplifiy comparing results between implementations/publications.

Each subfolder has two subsubfolders 3 and 7 which contain the `.jpg` files for the respective class of images. This is a common way of organizing datasets comprised of pictures. For the full `MNIST` dataset there are 10 subsubfolders, one for the images for each digit.


3. Explain how the **“pixel similarity”** approach to classifying digits works.

In the “pixel similarity” approach, we generate an archetype for each class we want to identify. In our case, we want to distinguish images of 3’s from images of 7’s. We define the archetypical 3 as the pixel-wise mean value of all 3’s in the training set. Analoguously for the 7’s. You can visualize the two archetypes and see that they are in fact blurred versions of the numbers they represent.
In order to tell if a previously unseen image is a 3 or a 7, we calculate its distance to the two archetypes (here: mean pixel-wise absolute difference). We say the new image is a 3 if its distance to the archetypical 3 is lower than two the archetypical 7.

5. *What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.*

```python
lst_in = range(10)
lst_out = [2*el for el in lst_in if el%2==1]
# is equivalent to:
lst_out = []
for el in lst_in:
   if el%2==1:
       lst_out.append(2*el)
```

5. What is a “rank 3 tensor”?


The rank of a tensor is the number of dimensions it has. An easy way to identify the rank is the number of indices you would need to reference a number within a tensor. A scalar can be represented as a tensor of rank 0 (no index), a vector can be represented as a tensor of rank 1 (one index, e.g., v[i]), a matrix can be represented as a tensor of rank 2 (two indices, e.g., a[i,j]), and a tensor of rank 3 is a cuboid or a “stack of matrices” (three indices, e.g., b[i,j,k]). In particular, the rank of a tensor is independent of its shape or dimensionality, e.g., a tensor of shape 2x2x2 and a tensor of shape 3x5x7 both have rank 3.
Note that the term “rank” has different meanings in the context of tensors and matrices (where it refers to the number of linearly independent column vectors).


6. What is the difference between tensor rank and shape?


Rank is the number of axes or dimensions in a tensor; shape is the size of each axis of a tensor.


- How do you get the rank from the shape?


The length of a tensor's shape is its rank.

So if we have the images of the 3 folder from the MINST_SAMPLE dataset in a tensor called `stacked_threes` and we find its shape like this.


```python
In [ ]: stacked_threes.shape
Out[ ]: torch.Size([6131, 28, 28])
```

`
We just need to find its length to know its rank. This is done as follows.
`

```python
In [ ]: len(stacked_threes.shape)
Out[ ]: 3
```

You can also get a tensor's rank directly with `ndim`.

```python
In [ ]: stacked_threes.ndim
Out[ ]: 3
```

7. What are RMSE and L1 norm?

Root mean square error (RMSE), also called the L2 norm, and mean absolute difference (MAE), also called the L1 norm, are two commonly used methods of measuring “distance”. Simple differences do not work because some difference are positive and others are negative, canceling each other out. Therefore, a function that focuses on the magnitudes of the differences is needed to properly measure distances. The simplest would be to add the absolute values of the differences, which is what MAE is. RMSE takes the mean of the square (makes everything positive) and then takes the square root (undoes squaring).

8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?

As loops are very slow in Python, it is best to represent the operations as array operations rather than looping through individual elements. If this can be done, then using NumPy or PyTorch will be thousands of times faster, as they use underlying C code which is much faster than pure Python. Even better, PyTorch allows you to run operations on GPU, which will have significant speedup if there are parallel operations that can be done.

9. Create a 3x3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom right 4 numbers.

```python
In [ ]: a = torch.Tensor(list(range(1,10))).view(3,3); print(a)
Out [ ]: tensor([[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]])
In [ ]: b = 2*a; print(b)
Out [ ]: tensor([[ 2.,  4.,  6.],
                 [ 8., 10., 12.],
                 [14., 16., 18.]])
In [ ]:  b[1:,1:]
Out []: tensor([[10., 12.],
                [16., 18.]])
```

## 



## Transfer learning


## Stochastic gradient descent (SGD)


## Data augmentation


## Weight decay


## Image classification


## Entity and word embeddings



