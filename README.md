### pycudamat
by Benjamin Labas

## Python Linear Algebra library utilizing CUDA

Allows off-loading heavy matrix computations like matrix multiplication to the GPU.

### System Requirements
* CUDA-capable GPU (see https://developer.nvidia.com/cuda-gpus)
* Python 3.6

### Dependencies
* CUDA 9.1
* CMake 3.9

### Build
* Generate build files: `cmake -A x64 -Bbuild`
* Build: `cmake --build build --target install --config Release`
* Install Python module: `cd python && python setup.py install`
* Run tests: `cd python && python -m unittest discover`

### Example
```python
$ python -q
>>> import pycudamat as pcm
>>>
>>> A = pcm.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> B = pcm.from_2d([[1, 2], [3, 4], [5, 6]])
>>>
>>> C = A.mult(B)
>>>
>>> C
Matrix=[
[ 22.000 28.000 ]
[ 49.000 64.000 ]
[ 76.000 100.000 ]
]
```
