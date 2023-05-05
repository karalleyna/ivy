# global
import abc
from typing import Optional, Union

# local
import ivy


class _ArrayWithCreationExperimental(abc.ABC):
    def eye_like(
        self: ivy.Array,
        /,
        *,
        k: int = 0,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.eye_like. This method simply wraps the
        function, and so the docstring for ivy.eye_like also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
        k
            index of the diagonal. A positive value refers to an upper diagonal,
            a negative value to a lower diagonal, and 0 to the main diagonal.
            Default: ``0``.
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data type
            must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If ``device`` is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array having the same shape as ``self`` and filled with ``ones``
            in diagonal ``k`` and ``zeros`` elsewhere.

        Examples
        --------
        >>> x = ivy.array([[2, 3, 8],[1, 2, 1]])
        >>> y = x.eye_like()
        >>> print(y)
        ivy.array([[1., 0., 0.],
                    0., 1., 0.]])
        """
        return ivy.eye_like(self._data, k=k, dtype=dtype, device=device, out=out)

    def trilu(self: ivy.Array, /, *, k: int = 0, out: Optional[ivy.Array] = None):
        """
        ivy.Array instance method variant of ivy.trilu. This method simply wraps the
        function, and so the docstring for ivy.trilu also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array having shape (..., M, N) and whose innermost two dimensions form
            MxN matrices.    *,
        k
            diagonal below which to zero elements. If k = 0, the diagonal is the main
            diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
            diagonal is above the main diagonal. Default: ``0``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the upper triangular part(s). The returned array must
            have the same shape and data type as ``self``. All elements below the
            specified diagonal k must be zeroed. The returned array should be allocated
            on the same device as ``self``.
        """
        return ivy.trilu(self._data, k=k, out=out)
