import ivy


def matrix_rank(M, tol=None):
    return ivy.matrix_rank(M, rtol=tol)
