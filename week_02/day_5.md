## Introduction

Linear Algebra is a branch of mathematics that allows you to concisely describe coordinates and interactions of planes in higher dimensions, as well as perform operations on them. 

Think of it as an extension of algebra into an arbitrary number of dimensions. Linear Algebra is about working on linear systems of equations. Rather than working with scalars, we work with matrices and vectors. This is particularly import to the study of computer science because vectors and matrices can be used to represent data of all forms - images, text, and of course, numerical values.

### Why Learn Linear Algebra?

<b>Machine Learning</b>: A lot of Machine Learning concepts are tied to linear algebra concepts. Some basic examples, PCA - eigenvalue, regression - matrix multiplication. As most ML techniques deal with high dimensional data, they are often times represented as matrices.

<b>Mathematical Modeling</b>: for example, if you want to capture behaviors (sales, engagement, etc.) in a mathematical model, you can use matrices to breakdown the samples into their own subgroups. This requires some basic matrix manipulation, such as atrix inversion, derivation, solving partial differential, or first order differential equations with matrices, for example. 

### 1.2 Scalars & Vectors

You'll see the terms scalar and vector throughout this course, so it's very important that we learn how to distinguish between the two. A scalar refers to the <b>magnitude</b> of an object. In contrast, a <i>vector</i> has <b>both</b> a magnitude and a <b>direction</b>. 

An intuitive example is with respect to distance. If you drive 50 miles north, then the scalar value is `50`. Now, the vector that would represent this could be something like `(50, N)` which indicates to us both the direction <i>and</i> the magnitude. 

### Importance

There are many reasons why the mathematics of Machine Learning is important and I’ll highlight some of them below:

1. Selecting the right algorithm which includes giving considerations to accuracy, training time, model complexity, number of parameters and number of features.

2. Choosing parameter settings and validation strategies.

3. Identifying underfitting and overfitting by understanding the Bias-Variance tradeoff.

4. Estimating the right confidence interval and uncertainty.

### Notation

&isin; refers to "element in". For example `2` &isin; `[1,2,3,4]` <br>

&real; refers to the set of all real numbers. 


## Vectors

As we mentioned in the previous section, a vector is typically an ordered tuple of numbers which have both a magnitude and direction. It's important to note that vectors are an <b>element</b> of a vector space. 

In the next section, we'll learn about matrices, which are a rectangular array of values. A vector is simply a one dimensional matrix. 

Sometimes what we're given, however, isn't a neat two value tuple consisting of a magnitude and direction value. Sometimes we're given something that resembles a list, and from there, we can create a tuple that describes the vector. 

With that said, we <i>can</i> represent a vector with a list, for example:  

``` python
A = [2.0, 3.0, 5.0]
```

From this vector we can then calculate the magnitude as we've done before:

```
nln.norm(A)
```
Getting us a value of approximately `6.16`. In many instances, vectors are also made with numpy arrays.

``` python
import numpy as np
A = np.array([2.0, 3.0, 4.0])
```


If we call the method `norm()` on this array, we get the same value, `6.16`.

### What is a vector space?

A vector space V is a set that contains all linear combinations of its elements. In other words, if you have a set A, the space vector V is a set of all combinations of the elements in the original set A. 

With that said, there are three properties that <b>every</b> vector space must follow:

1. Additive Closure: If vectors u and v &isin; V, then u + v &isin; V <br>
When we stated earlier that the vector space has all combinations of the elements in set A, one of the operations we meant by 'combinations' was <i>vector</i> addition. For example if we have two vectors in set A, let's say (4, 5) and (3, 1), then the vector space of A must have those two vectors, as well as the vector (4+3, 5+1), or `(7, 6)`. This has to be true for any two vectors in set A. 

2. Scalar Closure: If u &isin; V, then &alpha; u must &isin; &Nu; for any scalar &alpha; <br>
Recall that a scalar is a magnitude value with no direction, such as 5. For a vector space to be a vector space, that means for every vector in the original set A, that vector multiplied by any number (or constant or scalar) must be in the vector space V. 

3. Additive Identity: There exists a 0 &isin; V such that u + 0 = u for any u &isin; V
In other words, the vector space of set A must contain the zero vector.

4. Additive Associativity: If u, v, w &isin; V, then u + (v + w) = (u + v) + w
Regardless of the order in which you add multiple vectors, their results should be the same

5. Additive Inverse: If u &isin; V, then there exists a vector −u &isin; V so that u + (−u) = 0.
For example, if the vector (2, 3) &isin; V, then its additive inverse is (-2, -3) and must also be an element of the vector space V. 


The dimension of a vector space V is the cardinality. It's usually denoted as superscript, for example, &real;<sup>n</sup>. Let's break down what this looks like:

- &real;<sup>2</sup> refers to your typical x, y systems of equations. <br>

- &real;<sup>3</sup> adds an extra dimension, which means adding another variable, perhaps z.

### What is a subspace?

A vector subspace is a subset of a vector space. That subspace is <b>also</b> a vector space. If W is a linear subspace of V, then dim(W) must be &le; dim(V).

The easiest way to check whether it's a vector subspace is to check if it's closed under addition and scalar multiplication. Let's go over an example:

Let's show that the set V = {(x, y, z) | x, y, z &isin; &real; and x*x = z*z } is <b>not</b> a subspace of &real;<sup>3</sup>.

If V is actually a subspace of &real;<sup>3</sup>, that means it <b>must</b> follow all the properties listed in the beginning of this section. Recall that this is because all subspaces must also be vector spaces. 

Let's evaluate the first property that stays the following:

If vectors u and v &isin; V, then u + v &isin; V

Now, is this true of the set we've defined above? Absolutely not. (1, 1, 1) and (1, 1, -1) are both in V, but what about their sum, (1, 2, 0)? It's not! And because it's not, it does not follow the required properties of a vector space. Therefore, we can conluse that it's also not a subspace. 

### What is Linear Dependence? 

A set of vectors {v<sub>1</sub>,...,v<sub>n</sub>} is linearly independent if there are scalars c<sub>1</sub>...c<sub>n</sub> (which <b>aren't</b> all 0) such that the following is true:

c<sub>1</sub>v<sub>1</sub> + ... + c<sub>n</sub>v<sub>n</sub> = 0

#### Example

Let's say we have three vectors in a set: x<sub>1</sub> = (1, 1, 1), x<sub>2</sub> = (1, -1, 2), and x<sub>3</sub> = (3, 1, 4). 

These set of vectors are linear dependent because 2x<sub>1</sub> + x<sub>2</sub> - x<sub>3</sub> = 0. Why is this equal to zero? Again, because 2*(1,1,1) + 1(1,-1,2) - (3,1,4) = (2+1-3, 2-1-1, 2+2-4) = (0, 0, 0). If we can find some equation that satisfies a resultant of 0, it's considered linear dependent!

### What is Linear Independence? 

A set of vectors is considered linear dependent simply if they are not linear dependent! In other words, all the constants from the previous section should be equal to zero. c<sub>1</sub> = c<sub>2</sub> = ... = c<sub>n</sub> = 0

### What is a basis? 

Any linearly independent set of n vectors spans an n-dimensional space. This set of vectors is referred to as the basis of &real;<sup>n</sup>. 

#### Under vs Overdetermined Matrices

When `m < n`, the linear system is said to be <b>underdetermined</b>, e.g. there are fewer equations than unknowns. In this case, there are either no solutions or infinite solutions and a unique solution is not possible.

When `m > n`, the system may be <b>overdetermined</b>. In other words, there are more equations than unknowns. They system could be inconsistent, or some of the equations could be redundant. 

If some of the rows of an m x n matrix are linearly dependent, then the system is <b>reducible</b> and we get get rid of some of the rows. 

Lastly, if a matrix is square and its rows are linearly independent, the system has a unique solution and is considered <b>invertible</b>.

### What is a Norm? 

Remember the distance formula from the intro section? That's what a norm is, which is why that function we used from scipy was called `linalg.norm`. With that said, just to review, a norm just refers to the magnitude of a vector, and is denoted with ||u||. With numpy and scipy, we can do calculate the norm as follows: 

``` python
import numpy as np
from scipy import linalg
v = np.array([1,2])
linalg.norm(v)
```
The actual formula looks like: 

![alt text](https://github.com/lesley2958/lin-alg/blob/master/norm.png?raw=true "Logo Title Text 1")

## Matrices

A Matrix is a 2D array that stores real or complex numbers. You can think of them as multiple vectors in an array! You can use numpy to create matrices:

``` python
import numpy as np
matrix1 = np.matrix(
    [[0, 4],
     [2, 0]]
)
matrix2 = np.matrix(
    [[-1, 2],
     [1, -2]]
)
```

### Inverse

The matrix A is invertible if there exists a matrix A<sub>-1</sub> such that

A<sub>-1</sub>A = I and AA<sub>-1</sub> = I

Multiplying inverse matrix is like the division; it’s simply the reciprocal function of multiplication. Let’s say here is a square matrix A, then multiplying its inversion gives the identity matrix I.

We can get the inverse matrix with numpy: 

``` python
inverse = np.linalg.inv(matrix1)
```

### Identity Matrix

A Diagonal Matrix is an n x n matrix with 1s on the diagonal from the top left to the bottom right, such as 

``` 
[[ 1., 0., 0.],
[ 0., 1., 0.],
[ 0., 0., 1.]]
```
We can generate diagonal matrices with the `eye()` function in Python: 

``` python
np.eye(4)
```

When a matrix is multiplied by its inverse, the result is the identity matrix. It's important to note that only square matrices have inverses!


## Matrix Operations

### Addition

Matrix addition works very similarlty to normal addition. You simply add the corresponding spots together. 

``` python
matrix_sum = matrix1 + matrix2
```
And you'll get
``` 
matrix([[-1,  6],
        [ 3, -2]])
```

Visually, this a vector addition looks something like:

![alt text](https://github.com/lesley2958/lin-alg/blob/master/vector%20addition.png?raw=true "Logo Title Text 1")

### Multiplication

To multiply two matrices with numpy, you can use the `np.dot` method: 

``` python
np.dot(matrix2, matrix1)
```
Or, simply, you can do:

``` python
matrix_prod = matrix1 * matrix2
```

The dot product is an operation that takes two coordinate vectors of equal size and returns a single number. The result is calculated by multiplying corresponding entries and adding up those products. 

### Trace and Determinant

The trace of a matrix A is the sum of its diagonal elements. It's important because it's an invariant of a matrix under change of basis and it defines a matrix norm. 

In Python, this can be done with the `numpy` module: 

``` python
np.trace(matrix1)
```
which in this case is just `0`. 

The determinant of a matrix is defined to be the alternating sum of permutations of the elements of a matrix. The formula is as follows:

![alt text](https://github.com/lesley2958/lin-alg/blob/master/det.png?raw=true "Logo Title Text 1")

In python, you can use the following function: 

``` python
det = np.linalg.det(matrix1)
```
Which gets you `-7.99999` or about `-8`. Note that an n×n matrix A is invertible &iff; det(A) &ne; 0.


### Eigenvalues & Eigenvectors

Let A be an n x n matrix. The number &lambda; is an eigenvalue of A if there exists a non-zero vector C such that

Av = &lambda;v

In this case, vector v is called an eigenvector of A corresponding to &lambda;. You can use numpy to calculate the eigenvectors of a matrix: 

``` python
eigvals = np.linalg.eigvals(matrix)
```

Note that eigenvectors do not change direction in the transformation of the matrix.

### Solving Systems of Equations

Consider a set of m linear equations in n unknowns:

![alt text](https://github.com/ByteAcademyCo/stats-programmers/blob/master/system1.png?raw=true "Logo Title Text 1")

We can let:

![alt text](https://github.com/ByteAcademyCo/stats-programmers/blob/master/system2.png?raw=true "Logo Title Text 1")

And re-write the system: 

```
Ax = b
```
This reduces the problem to a matrix equation and now we can solve the system to find A<sup>−1</sup>.

#### Systems of Equations with Python

Using numpy, we can solve systems of equations in Python:

``` python 
import numpy as np
```

Each equation in a system can be represented with matrices. For example, if we have the equation `3x - 9y = -42`, it can be represented as `[3, -9]` and `[-42]`. If we add another equation to the mix, for example,  `2x + 4y = 2`, we can merge it with the previous equation to get `[[3, -9], [2, 4]]` and `[-42, 2]`. Now let's solve for the x and y values.

Now, let's put these equations into numpy arrays:

``` python
A = np.array([ [3,-9], [2,4] ])
b = np.array([-42,2])
```

Now, we can use the `linalg.solve()` function to solve the x and y values. Note that these values will be
```
z = np.linalg.solve(A,b)
```

This gets us: 

``` 
array([-5.,  3.])
```
which means `x = -5` and `y = 3`. 


## Matrix Types

### Under vs Overdetermined Matrices

When `m < n`, the linear system is said to be <b>underdetermined</b>, e.g. there are fewer equations than unknowns. In this case, there are either no solutions or infinite solutions and a unique solution is not possible.

When `m > n`, the system may be <b>overdetermined</b>. In other words, there are more equations than unknowns. They system could be inconsistent, or some of the equations could be redundant. 

### Row, Column, and Null Space 

The <b>column space</b> C(A) of a matrix A (sometimes called the range of a matrix) is the span (set of all possible linear combinations) of its column vectors.

The <b>row space</b> of an m x n matrix, A, denoted by R(A) is the set of all linear combinations of the row vectors of A.

The <b>null space</b> of an m x n matrix, A, denoted by null(A) is the set of all solutions, x, of the equation Ax = 0<sub>m</sub>.

### Rank

The rank of a matrix A is the dimension of its column space - and - the dimension of its row space. These are equal for any matrix. Rank can be thought of as a measure of non-degeneracy of a system of linear equations, in that it is the dimension of the image of the linear transformation determined by A.

### Kernels & Images

#### Images 

The image of a function consists of all the values the function takes in its codomain.

#### Kernels

The kernel of a matrix A is the dimension of the space mapped to zero under the linear transformation that A represents. The dimension of the kernel of a linear transformation is called the nullity.

### Matrix Norms

We can extend the notion of a norm of a vector to a norm of a matrix. Matrix norms are used in determining the condition of a matrix. There are many matrix norms, but three of the most common are so called ‘p’ norms, and they are based on p-norms of vectors.

So, for an n-dimensional vector v and for 1 &le; p &le; &infin;, we have the following formula:

![alt text](https://github.com/lesley2958/lin-alg/blob/master/p%20norm.png?raw=true "Logo Title Text 1")

And for p = &infin;:

![alt text](https://github.com/lesley2958/lin-alg/blob/master/infty%20norm.png?raw=true "Logo Title Text 1")

The corresponding matrix norms are:

![alt text](https://github.com/lesley2958/lin-alg/blob/master/matrix%20norms.png?raw=true "Logo Title Text 1")
