# einsum for xtensor
This project implements the numpy `einsum` function for the xtensor project. (It borrows heavily from the numpy implementation.) Currently, the syntax looks like this
```c++
enum labels: int {
  _ = 0,
  I,
  J,
  K
}

auto x = einsum<subscripts<I,J,_>,subscripts<J,K,_>>(_).eval<subscripts<K,_,I>>(y,z)
auto w = einsum<subscripts<I,J,_>,subscripts<J,K,_>>(_).eval<implicit_out>(y,z)
```
for the equivalent numpy statements
```python
x = np.einsum('ij,jk...->k...i', y, z)
w = np.einsum('ij,jk...', y, z)
```
Please include the `main/include` directory to test this in your project.

To run the tests, set the `$EINSUM_TEST_HOME` environment variable to the full path to the `test` directory on your machine and then run `make && ./test/main`.
