# 中文学习注释：MAPS/Ceviche 第三方核心依赖阅读辅助。未改变上游算法、参数或执行逻辑。
# 模块定位：thirdparty/MAPS_old/thirdparty/ceviche/jacobians.py。重点追踪 Maxwell 离散、线性系统、边界/源项、伴随导数及结构参数化。
# 注意：这里的注释是学习说明，不代表上游项目作者原始注释。

import autograd.numpy as npa

from autograd.core import make_vjp, make_jvp
from autograd.wrap_util import unary_to_nary
from autograd.extend import vspace

from .utils import get_value, get_shape, get_value_arr, float_2_array


"""
This file provides wrappers to autograd that compute jacobians.  
The only function you'll want to use in your code is `jacobian`, 
where you can specify the mode of differentiation (reverse, forward, or numerical)
"""

# [中文注释] 构造离散导数/Jacobian 算子，为场求解或伴随梯度提供数值基础。
def jacobian(fun, argnum=0, mode='reverse', step_size=1e-6):
    """ Computes jacobian of `fun` with respect to argument number `argnum` using automatic differentiation """

    if mode == 'reverse':
        return jacobian_reverse(fun, argnum)
    elif mode == 'forward':
        return jacobian_forward(fun, argnum)
    elif mode == 'numerical':
        return jacobian_numerical(fun, argnum, step_size=step_size)
    else:
        raise ValueError("'mode' kwarg must be either 'reverse' or 'forward' or 'numerical', given {}".format(mode))


@unary_to_nary
# [中文注释] 构造离散导数/Jacobian 算子，为场求解或伴随梯度提供数值基础。
def jacobian_reverse(fun, x):
    """ Compute jacobian of fun with respect to x using reverse mode differentiation"""
    vjp, ans = make_vjp(fun, x)
    grads = map(vjp, vspace(ans).standard_basis())
    m, n = _jac_shape(x, ans)
    return npa.reshape(npa.stack(grads), (n, m))


@unary_to_nary
# [中文注释] 构造离散导数/Jacobian 算子，为场求解或伴随梯度提供数值基础。
def jacobian_forward(fun, x):
    """ Compute jacobian of fun with respect to x using forward mode differentiation"""
    jvp = make_jvp(fun, x)
    # ans = fun(x)
    val_grad = map(lambda b: jvp(b), vspace(x).standard_basis())
    vals, grads = zip(*val_grad)
    ans = npa.zeros((list(vals)[0].size,))  # fake answer so that dont have to compute it twice
    m, n = _jac_shape(x, ans)
    if _iscomplex(x):
        grads_real = npa.array(grads[::2])
        grads_imag = npa.array(grads[1::2])
        grads = grads_real - 1j * grads_imag
    return npa.reshape(npa.stack(grads), (m, n)).T


@unary_to_nary
# [中文注释] 构造离散导数/Jacobian 算子，为场求解或伴随梯度提供数值基础。
def jacobian_numerical(fn, x, step_size=1e-7):
    """ numerically differentiate `fn` w.r.t. its argument `x` """
    in_array = float_2_array(x).flatten()
    out_array = float_2_array(fn(x)).flatten()

    m = in_array.size
    n = out_array.size
    shape = (n, m)
    jacobian = npa.zeros(shape)

    for i in range(m):
        input_i = in_array.copy()
        input_i[i] += step_size
        arg_i = input_i.reshape(in_array.shape)
        output_i = fn(arg_i).flatten()
        grad_i = (output_i - out_array) / step_size
        jacobian[:, i] = get_value_arr(get_value(grad_i))  # need to convert both the grad_i array and its contents to actual data.

    return jacobian


# [中文注释] 实现 _jac_shape；结合调用处核对物理量、网格维度、边界条件和单位。
def _jac_shape(x, ans):
    """ computes the shape of the jacobian where function has input x and output ans """
    m = float_2_array(x).size
    n = float_2_array(ans).size
    return (m, n)


# [中文注释] 实现 _iscomplex；结合调用处核对物理量、网格维度、边界条件和单位。
def _iscomplex(x):
    """ Checks if x is complex-valued or not """
    if isinstance(x, npa.ndarray):
        if x.dtype == npa.complex128:
            return True
    if isinstance(x, complex):
        return True
    return False


if __name__ == '__main__':

    """ Some simple test """

    N = 3
    M = 2
    A = npa.random.random((N,M))
    B = npa.random.random((N,M))
    print('A = \n', A)

    # [中文注释] 实现 fn；结合调用处核对物理量、网格维度、边界条件和单位。
    def fn(x, b):
        return A @ x + B @ b

    x0 = npa.random.random((M,))
    b0 = npa.random.random((M,))    
    print('Jac_rev = \n', jacobian(fn, argnum=0, mode='reverse')(x0, b0))
    print('Jac_for = \n', jacobian(fn, argnum=0, mode='forward')(x0, b0))
    print('Jac_num = \n', jacobian(fn, argnum=0, mode='numerical')(x0, b0))

    print('B = \n', B)
    print('Jac_rev = \n', jacobian(fn, argnum=1, mode='reverse')(x0, b0))
    print('Jac_for = \n', jacobian(fn, argnum=1, mode='forward')(x0, b0))
    print('Jac_num = \n', jacobian(fn, argnum=1, mode='numerical')(x0, b0))

