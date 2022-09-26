# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# matrix_rank
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        min_value=-1e05,
        max_value=1e05,
    ),
    rtol=st.floats(allow_nan=False, allow_infinity=False) | st.just(None),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.linalg.matrix_rank"
    ),
)
def test_jax_numpy_matrix_rank(
    dtype_and_x, rtol, as_variable, native_array, num_positional_args, fw
):
    dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=dtype[0])

    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        native_array_flags=native_array,
        num_positional_args=num_positional_args,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.linalg.matrix_rank",
        M=x,
        tol=rtol,
    )
