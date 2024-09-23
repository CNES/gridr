from scipy import signal
from scipy.signal import _signaltools
import numpy as np
import math

def oaconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using the overlap-add method.

    Convolve `in1` and `in2` using the overlap-add method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    and generally much faster than `fftconvolve` when one array is much
    larger than the other, but can be slower when only a few output values are
    needed or when the arrays are very similar in shape, and can only
    output float arrays (int or object array inputs will be cast to float).

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    convolve : Uses the direct convolution or FFT convolution algorithm
               depending on which is faster.
    fftconvolve : An implementation of convolution using FFT.

    Notes
    -----
    .. versionadded:: 1.4.0

    References
    ----------
    .. [1] Wikipedia, "Overlap-add_method".
           https://en.wikipedia.org/wiki/Overlap-add_method
    .. [2] Richard G. Lyons. Understanding Digital Signal Processing,
           Third Edition, 2011. Chapter 13.10.
           ISBN 13: 978-0137-02741-5

    Examples
    --------
    Convolve a 100,000 sample signal with a 512-sample filter.

    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> sig = rng.standard_normal(100000)
    >>> filt = signal.firwin(512, 0.01)
    >>> fsig = signal.oaconvolve(sig, filt)

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('White noise')
    >>> ax_mag.plot(fsig)
    >>> ax_mag.set_title('Filtered noise')
    >>> fig.tight_layout()
    >>> fig.show()

    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])
    elif in1.shape == in2.shape:  # Equivalent to fftconvolve
        return _signaltools.fftconvolve(in1, in2, mode=mode, axes=axes)

    in1, in2, axes = _signaltools._init_freq_conv_axes(in1, in2, mode, axes,
                                          sorted_axes=True)

    s1 = in1.shape
    s2 = in2.shape

    print("s1, s2, axes", s1, s2, axes)
    
    if not axes:
        ret = in1 * in2
        return _signaltools._apply_conv_mode(ret, s1, s2, mode, axes)

    # Calculate this now since in1 is changed later
    shape_final = [None if i not in axes else
                   s1[i] + s2[i] - 1 for i in range(in1.ndim)]
    
    print('shape_finale', shape_final)

    # Calculate the block sizes for the output, steps, first and second inputs.
    # It is simpler to calculate them all together than doing them in separate
    # loops due to all the special cases that need to be handled.
    optimal_sizes = [(-1, -1, s1[i], s2[i]) if i not in axes else
                     _signaltools._calc_oa_lens(s1[i], s2[i]) for i in range(in1.ndim)]
    
    print('optimal sizes', list(optimal_sizes))
    block_size, overlaps, \
        in1_step, in2_step = zip(*optimal_sizes)

    # Fall back to fftconvolve if there is only one block in every dimension.
    if in1_step == s1 and in2_step == s2:
        return _signaltools.fftconvolve(in1, in2, mode=mode, axes=axes)

    # Figure out the number of steps and padding.
    # This would get too complicated in a list comprehension.
    nsteps1 = []
    nsteps2 = []
    pad_size1 = []
    pad_size2 = []
    for i in range(in1.ndim):
        if i not in axes:
            pad_size1 += [(0, 0)]
            pad_size2 += [(0, 0)]
            continue

        if s1[i] > in1_step[i]:
            curnstep1 = math.ceil((s1[i]+1)/in1_step[i])
            if (block_size[i] - overlaps[i])*curnstep1 < shape_final[i]:
                curnstep1 += 1

            curpad1 = curnstep1*in1_step[i] - s1[i]
        else:
            curnstep1 = 1
            curpad1 = 0

        if s2[i] > in2_step[i]:
            curnstep2 = math.ceil((s2[i]+1)/in2_step[i])
            if (block_size[i] - overlaps[i])*curnstep2 < shape_final[i]:
                curnstep2 += 1

            curpad2 = curnstep2*in2_step[i] - s2[i]
        else:
            curnstep2 = 1
            curpad2 = 0

        nsteps1 += [curnstep1]
        nsteps2 += [curnstep2]
        pad_size1 += [(0, curpad1)]
        pad_size2 += [(0, curpad2)]

    # Pad the array to a size that can be reshaped to the desired shape
    # if necessary.
    if not all(curpad == (0, 0) for curpad in pad_size1):
        in1 = np.pad(in1, pad_size1, mode='constant', constant_values=0)

    if not all(curpad == (0, 0) for curpad in pad_size2):
        in2 = np.pad(in2, pad_size2, mode='constant', constant_values=0)

    # Reshape the overlap-add parts to input block sizes.
    split_axes = [iax+i for i, iax in enumerate(axes)]
    fft_axes = [iax+1 for iax in split_axes]

    # We need to put each new dimension before the corresponding dimension
    # being reshaped in order to get the data in the right layout at the end.
    reshape_size1 = list(in1_step)
    reshape_size2 = list(in2_step)
    for i, iax in enumerate(split_axes):
        reshape_size1.insert(iax, nsteps1[i])
        reshape_size2.insert(iax, nsteps2[i])

    in1 = in1.reshape(*reshape_size1)
    in2 = in2.reshape(*reshape_size2)

    # Do the convolution.
    fft_shape = [block_size[i] for i in axes]
    ret = _signaltools._freq_domain_conv(in1, in2, fft_axes, fft_shape, calc_fast_len=False)

    # Do the overlap-add.
    for ax, ax_fft, ax_split in zip(axes, fft_axes, split_axes):
        overlap = overlaps[ax]
        if overlap is None:
            continue

        ret, overpart = np.split(ret, [-overlap], ax_fft)
        overpart = np.split(overpart, [-1], ax_split)[0]

        ret_overpart = np.split(ret, [overlap], ax_fft)[0]
        ret_overpart = np.split(ret_overpart, [1], ax_split)[1]
        ret_overpart += overpart

    # Reshape back to the correct dimensionality.
    shape_ret = [ret.shape[i] if i not in fft_axes else
                 ret.shape[i]*ret.shape[i-1]
                 for i in range(ret.ndim) if i not in split_axes]
    ret = ret.reshape(*shape_ret)

    # Slice to the correct size.
    slice_final = tuple([slice(islice) for islice in shape_final])
    ret = ret[slice_final]

    return _signaltools._apply_conv_mode(ret, s1, s2, mode, axes)

def calc_oa_chunk(s1, s2, smax):
   """
   Inspired from scipy.signal._signaltools._calc_oa_lens
   """
    # Set up the arguments for the conventional FFT approach.
    fallback = (s1+s2-1, None, s1, s2)

    # Use conventional FFT convolve if sizes are same.
    if s1 == s2 or s1 == 1 or s2 == 1:
        return fallback

    if s2 > s1:
        s1, s2 = s2, s1
        swapped = True
    else:
        swapped = False

    # There cannot be a useful block size if s2 is more than half of s1.
    if s2 >= s1/2:
        return fallback


def _calc_oa_lens(s1, s2):
    """Calculate the optimal FFT lengths for overlapp-add convolution.

    The calculation is done for a single dimension.

    Parameters
    ----------
    s1 : int
        Size of the dimension for the first array.
    s2 : int
        Size of the dimension for the second array.

    Returns
    -------
    block_size : int
        The size of the FFT blocks.
    overlap : int
        The amount of overlap between two blocks.
    in1_step : int
        The size of each step for the first array.
    in2_step : int
        The size of each step for the first array.

    """
    # Set up the arguments for the conventional FFT approach.
    fallback = (s1+s2-1, None, s1, s2)

    # Use conventional FFT convolve if sizes are same.
    if s1 == s2 or s1 == 1 or s2 == 1:
        return fallback

    if s2 > s1:
        s1, s2 = s2, s1
        swapped = True
    else:
        swapped = False

    # There cannot be a useful block size if s2 is more than half of s1.
    if s2 >= s1/2:
        return fallback

    # Derivation of optimal block length
    # For original formula see:
    # https://en.wikipedia.org/wiki/Overlap-add_method
    #
    # Formula:
    # K = overlap = s2-1
    # N = block_size
    # C = complexity
    # e = exponential, exp(1)
    #
    # C = (N*(log2(N)+1))/(N-K)
    # C = (N*log2(2N))/(N-K)
    # C = N/(N-K) * log2(2N)
    # C1 = N/(N-K)
    # C2 = log2(2N) = ln(2N)/ln(2)
    #
    # dC1/dN = (1*(N-K)-N)/(N-K)^2 = -K/(N-K)^2
    # dC2/dN = 2/(2*N*ln(2)) = 1/(N*ln(2))
    #
    # dC/dN = dC1/dN*C2 + dC2/dN*C1
    # dC/dN = -K*ln(2N)/(ln(2)*(N-K)^2) + N/(N*ln(2)*(N-K))
    # dC/dN = -K*ln(2N)/(ln(2)*(N-K)^2) + 1/(ln(2)*(N-K))
    # dC/dN = -K*ln(2N)/(ln(2)*(N-K)^2) + (N-K)/(ln(2)*(N-K)^2)
    # dC/dN = (-K*ln(2N) + (N-K)/(ln(2)*(N-K)^2)
    # dC/dN = (N - K*ln(2N) - K)/(ln(2)*(N-K)^2)
    #
    # Solve for minimum, where dC/dN = 0
    # 0 = (N - K*ln(2N) - K)/(ln(2)*(N-K)^2)
    # 0 * ln(2)*(N-K)^2 = N - K*ln(2N) - K
    # 0 = N - K*ln(2N) - K
    # 0 = N - K*(ln(2N) + 1)
    # 0 = N - K*ln(2Ne)
    # N = K*ln(2Ne)
    # N/K = ln(2Ne)
    #
    # e^(N/K) = e^ln(2Ne)
    # e^(N/K) = 2Ne
    # 1/e^(N/K) = 1/(2*N*e)
    # e^(N/-K) = 1/(2*N*e)
    # e^(N/-K) = K/N*1/(2*K*e)
    # N/K*e^(N/-K) = 1/(2*e*K)
    # N/-K*e^(N/-K) = -1/(2*e*K)
    #
    # Using Lambert W function
    # https://en.wikipedia.org/wiki/Lambert_W_function
    # x = W(y) It is the solution to y = x*e^x
    # x = N/-K
    # y = -1/(2*e*K)
    #
    # N/-K = W(-1/(2*e*K))
    #
    # N = -K*W(-1/(2*e*K))
    overlap = s2-1
    opt_size = -overlap*lambertw(-1/(2*math.e*overlap), k=-1).real
    block_size = sp_fft.next_fast_len(math.ceil(opt_size))

    # Use conventional FFT convolve if there is only going to be one block.
    if block_size >= s1:
        return fallback

    if not swapped:
        in1_step = block_size-s2+1
        in2_step = s2
    else:
        in1_step = s2
        in2_step = block_size-s2+1

    return block_size, overlap, in1_step, in2_step


def test0():
    nrow, ncol = 50, 60
    kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
    image = np.arange(nrow*ncol, dtype=np.float32).reshape((nrow, ncol))
    
    kernel3 = np.array([kernel, kernel, kernel])
    image3 = np.array([image, image, image])
    
    # TODO : verifier que les axes fournis concernent bien que 2 dimensions
    out = oaconvolve(image3, kernel3, axes=[1,2])
    print(out.shape)
    print(out[:,1:3,1:3])
    
if __name__ == '__main__':
    test0()