# global
from typing import Union, Optional

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.utils.exceptions import handle_exceptions
from ivy.func_wrapper import (
    handle_array_function,
    handle_nestable,
    integer_arrays_to_float,
    to_native_arrays_and_back,
    handle_array_like_without_promotion,
    handle_out_argument,
)


@handle_out_argument
@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def logit(
    x: Union[float, int, ivy.Array],
    /,
    *,
    eps: Optional[float] = None,
    out: Optional["ivy.Array"] = None,
) -> ivy.Array:
    """
    Computes the logit of x, i.e. logit(x) = log(x / (1 - x)).

    Parameters
    ----------
    x
        Input data.
    eps
        When eps is None the function outpus NaN where x < 0 or x > 1.
        and inf or -inf where x = 1 or x = 0, respectively.
        Otherwise if eps is defined, x is clamped to [eps, 1 - eps]
    out
        Optional output array.

    Returns
    -------
    ret
        Array containing elementwise logits of x.

    Examples
    --------
    >>> x = ivy.array([1, 0, 0.9])
    >>> z = ivy.logit(x)
    >>> print(z)
    ivy.array([       inf,       -inf, 2.19722438])

    >>> x = ivy.array([1, 2, -0.9])
    >>> z = ivy.logit(x, eps=0.2)
    >>> print(z)
    ivy.array([ 1.38629448,  1.38629448, -1.38629436])

    """
    return current_backend(x).logit(x, eps=eps, out=out)


@handle_out_argument
@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def prelu(
    x: Union[ivy.NativeArray, ivy.Array],
    slope: Union[float, ivy.NativeArray, ivy.Array],
    /,
    *,
    out: Optional["ivy.Array"] = None,
) -> ivy.Array:
    """
    Prelu takes input data (Array) and slope array as input,
    and produces one output data (array) where the function
    f(x) = slope * x for x < 0, f(x) = x for x >= 0., is applied
    to the data array elementwise. This operator supports unidirectional
    broadcasting (array slope should be unidirectional broadcastable to
    input tensor X);

    Parameters
    ----------
    x
        Input Array.
    slope
        Slope Array. The shape of slope can be smaller then first input X;
        if so, its shape must be unidirectional broadcastable to X.
    out
        Optional output array.

    Returns
    -------
    ret
         Array containing Parametrized relu values.
    """
    try:
        return ivy.where(x > 0, x, x * slope, out=out)
    except ivy.utils.exceptions.IvyError(
        f"The shape {slope.shape} is not Unidirectional Broadcastable\n"
        f"as per ONNX standards"
    ) as IvyException:
        if len(slope.shape) == 1:
            dim = slope.shape[0]
            new_shape = []
            n = 0
            for d in x.shape:
                if d == dim:
                    new_shape.append(d)
                    n += 1
                else:
                    new_shape.append(d)
            if n == 1:
                xs = x * slope.reshape(tuple(new_shape), out=out)
                return ivy.where(x > 0, x, xs, out=out)
        raise IvyException


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def thresholded_relu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    threshold: Optional[Union[int, float]] = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the rectified linear unit function with custom threshold.

    Parameters
    ----------
    x
        input array
    threshold
        threshold value above which the activation is linear. Default: ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the rectified linear unit activation of each element in
        ``x``. with custom ``threshold``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1., 0., 1.])
    >>> y = ivy.thresholded_relu(x, threshold=0.5)
    >>> print(y)
    ivy.array([0.,  0. ,  1.])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.thresholded_relu(x, threshold=1, out = y)
    >>> print(y)
    ivy.array([ 1.5,  0., 0.])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.2, 0.6]))
    >>> x = ivy.thresholded_relu(x, threshold=0.5)
    >>> print(x)
    {
        a: ivy.array([1., 0.]),
        b: ivy.array([0., 0.6])
    }
    """
    return current_backend(x).thresholded_relu(x, threshold=threshold, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def relu6(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Applies the rectified linear unit 6 function element-wise.

    Parameters
    ----------
    x
        input array
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the rectified linear unit 6 activation of each element in
        ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> y = ivy.relu6(x)
    >>> print(y)
    ivy.array([0., 0., 1., 2., 3., 4., 5., 6., 6.])

    >>> x = ivy.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> y = ivy.zeros(9)
    >>> ivy.relu6(x, out = y)
    >>> print(y)
    ivy.array([0., 0., 1., 2., 3., 4., 5., 6., 6.])

    With :class:`ivy.Container` input:

    >>> x = {
                a: ivy.array([-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
                b: ivy.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
            }
    >>> x = ivy.relu6(x, out=x)
    >>> print(x)
    {
    a: ivy.array([0., 0., 0., 0., 1., 2., 3., 4., 5.]),
    b: ivy.array([1., 2., 3., 4., 5., 6., 6., 6., 6.])
    }
    """
    return current_backend(x).relu6(x, out=out)


@handle_out_argument
@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def batch_norm(
    x: Union[ivy.NativeArray, ivy.Array],
    mean: Union[ivy.NativeArray, ivy.Array],
    variance: Union[ivy.NativeArray, ivy.Array],
    /,
    *,
    offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    training: bool = False,
    eps: float = 1e-5,
) -> ivy.Array:
    """
    Applies batch normalization to the input array.

    Parameters
    ----------
    x
        Input array of shape (N,C,S), where N is the batch dimension, C is the feature
        dimension and S corresponds to the following spatial dimensions.
    mean
        A mean array for the input's normalization.
    variance
        A variance array for the input's normalization.
    offset
        An offset array. If present, will be added to the normalized input.
    scale
        A scale array. If present, the scale is applied to the normalized input.
    training
        If true, calculate and use the mean and variance of `x`. Otherwise, use the
        provided `mean` and `variance`.
    eps
        A small float number to avoid dividing by 0.

    Returns
    -------
    ret
         Array containing the normalized, scaled, offset values.
    """
    return current_backend(x).batch_norm(
        x,
        mean,
        variance,
        scale=scale,
        offset=offset,
        training=training,
        eps=eps,
    )


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def sigmoid(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Applies the sigmoid function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        an array containing the sigmoid activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.sigmoid()
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])


    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([[0.214, 0.978, 0.891], [0.846,0.985,0.001]] )
    """
    return current_backend(x).sigmoid(x, out=out)


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def hard_sigmoid(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Applies the sigmoid function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        an array containing the sigmoid activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.sigmoid()
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])


    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([[0.214, 0.978, 0.891], [0.846,0.985,0.001]] )
    """
    return current_backend(x).hard_sigmoid(x, out=out)


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def selu(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Applies the sigmoid function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        an array containing the sigmoid activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.sigmoid()
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])


    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([[0.214, 0.978, 0.891], [0.846,0.985,0.001]] )
    """
    return current_backend(x).selu(x, out=out)


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def hard_tanh(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Applies the sigmoid function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        an array containing the sigmoid activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.sigmoid()
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])


    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([[0.214, 0.978, 0.891], [0.846,0.985,0.001]] )
    """
    return current_backend(x).hard_tanh(x, out=out)


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def log_sigmoid(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Applies the sigmoid function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        an array containing the sigmoid activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.sigmoid()
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])


    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([[0.214, 0.978, 0.891], [0.846,0.985,0.001]] )
    """
    return current_backend(x).log_sigmoid(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def softplus(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the softplus function element-wise.

    Parameters
    ----------
    x
        input array.
    beta
        The beta value for the softplus formation. Default: ``None``.
    threshold
        values above this revert to a linear function. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the softplus activation of each element in ``x``.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softplus(x)
    >>> print(y)
    ivy.array([0.535,0.42])

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softplus(x, beta=0.5)
    >>> print(y)
    ivy.array([1.22, 1.09])

    >>> x = ivy.array([1., 2., 3.])
    >>> y = ivy.softplus(x, threshold=2)
    >>> print(y)
    ivy.array([1.31, 2.13, 3.  ])

    """
    return current_backend(x).softplus(x, beta=beta, threshold=threshold, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def softsign(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the softsign function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the softplus activation of each element in ``x``.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softsign(x)
    >>> print(y)
    ivy.array([0.535,0.42])

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softsign(x, beta=0.5)
    >>> print(y)
    ivy.array([1.22, 1.09])

    >>> x = ivy.array([1., 2., 3.])
    >>> y = ivy.softsign(x, threshold=2)
    >>> print(y)
    ivy.array([1.31, 2.13, 3.  ])

    """
    return current_backend(x).softsign(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def silu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the softsign function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the softplus activation of each element in ``x``.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softsign(x)
    >>> print(y)
    ivy.array([0.535,0.42])

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softsign(x, beta=0.5)
    >>> print(y)
    ivy.array([1.22, 1.09])

    >>> x = ivy.array([1., 2., 3.])
    >>> y = ivy.softsign(x, threshold=2)
    >>> print(y)
    ivy.array([1.31, 2.13, 3.  ])

    """
    return current_backend(x).silu(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def hard_silu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the softsign function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the softplus activation of each element in ``x``.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softsign(x)
    >>> print(y)
    ivy.array([0.535,0.42])

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softsign(x, beta=0.5)
    >>> print(y)
    ivy.array([1.22, 1.09])

    >>> x = ivy.array([1., 2., 3.])
    >>> y = ivy.softsign(x, threshold=2)
    >>> print(y)
    ivy.array([1.31, 2.13, 3.  ])

    """
    return current_backend(x).hard_silu(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def leaky_relu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: float = 0.2,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the leaky rectified linear unit function element-wise.

    Parameters
    ----------
    x
        Input array.
    alpha
        Negative slope for ReLU.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with leaky relu applied element-wise.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0.39, -0.85])
    >>> y = ivy.leaky_relu(x)
    >>> print(y)
    ivy.array([ 0.39, -0.17])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.leaky_relu(x, out=y)
    >>> print(y)
    ivy.array([ 1.5 ,  0.7 , -0.48])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> ivy.leaky_relu(x, out=x)
    >>> print(x)
    ivy.array([[ 1.1 ,  2.2 ,  3.3 ],
       [-0.88, -1.1 , -1.32]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.0, -1.2]), b=ivy.array([0.4, -0.2]))
    >>> x = ivy.leaky_relu(x, out=x)
    >>> print(x)
    {
        a: ivy.array([0., -0.24000001]),
        b: ivy.array([0.40000001, -0.04])
    }
    """
    return current_backend(x).leaky_relu(x, alpha=alpha, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def elu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: float = 1.0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the leaky rectified linear unit function element-wise.

    Parameters
    ----------
    x
        Input array.
    alpha
        Negative slope for ReLU.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with leaky relu applied element-wise.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0.39, -0.85])
    >>> y = ivy.leaky_relu(x)
    >>> print(y)
    ivy.array([ 0.39, -0.17])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.leaky_relu(x, out=y)
    >>> print(y)
    ivy.array([ 1.5 ,  0.7 , -0.48])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> ivy.leaky_relu(x, out=x)
    >>> print(x)
    ivy.array([[ 1.1 ,  2.2 ,  3.3 ],
       [-0.88, -1.1 , -1.32]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.0, -1.2]), b=ivy.array([0.4, -0.2]))
    >>> x = ivy.leaky_relu(x, out=x)
    >>> print(x)
    {
        a: ivy.array([0., -0.24000001]),
        b: ivy.array([0.40000001, -0.04])
    }
    """
    return current_backend(x).elu(x, alpha=alpha, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def celu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: float = 1.0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the leaky rectified linear unit function element-wise.

    Parameters
    ----------
    x
        Input array.
    alpha
        Negative slope for ReLU.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with leaky relu applied element-wise.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0.39, -0.85])
    >>> y = ivy.leaky_relu(x)
    >>> print(y)
    ivy.array([ 0.39, -0.17])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.leaky_relu(x, out=y)
    >>> print(y)
    ivy.array([ 1.5 ,  0.7 , -0.48])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> ivy.leaky_relu(x, out=x)
    >>> print(x)
    ivy.array([[ 1.1 ,  2.2 ,  3.3 ],
       [-0.88, -1.1 , -1.32]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.0, -1.2]), b=ivy.array([0.4, -0.2]))
    >>> x = ivy.leaky_relu(x, out=x)
    >>> print(x)
    {
        a: ivy.array([0., -0.24000001]),
        b: ivy.array([0.40000001, -0.04])
    }
    """
    return current_backend(x).celu(x, alpha=alpha, out=out)


@handle_out_argument
@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def glu(
    x: Union[ivy.NativeArray, ivy.Array],
    /,
    *,
    axis: int = -1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Applies batch normalization to the input array.

    Parameters
    ----------
    x
        Input array of shape (N,C,S), where N is the batch dimension, C is the feature
        dimension and S corresponds to the following spatial dimensions.
    mean
        A mean array for the input's normalization.

    Returns
    -------
    ret
         Array containing the normalized, scaled, offset values.
    """
    return current_backend(x).glu(x, axis=axis, out=out)
