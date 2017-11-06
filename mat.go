// The mat package provides an implementation of matrices and vectors that is
// completely immutable and focused on exposing a nice API rather than going for
// high performance.
package mat

import (
	"bytes"
	"fmt"
	"sync"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
)

// Matrix is optimized for dense matrices.
type Matrix struct {
	rows, cols int
	data       []float64
}

// New returns a new matrix with the given dimensions and where all cells are zero.
func New(rows, cols int) Matrix {
	return Matrix{rows, cols, make([]float64, rows*cols)}
}

// FromSlice returns a new Matrix with the contents of the given slice.
func FromSlice(rows, cols int, data []float64) Matrix {
	if len(data) != rows*cols {
		panicf("missmatched dimensions and data: %d x %d != %d", rows, cols, len(data))
	}

	c := make([]float64, len(data))
	copy(c, data)
	return Matrix{rows, cols, c}
}

func (m Matrix) ToSlice() []float64 { return m.data }

// FromFunc returns a new Matrix with the contents initialized by calling f.
func FromFunc(rows, cols int, f func(i, j int) float64) Matrix {
	m := Matrix{rows, cols, make([]float64, rows*cols)}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.set(i, j, f(i, j))
		}
	}
	return m
}

func (m Matrix) String() string {
	buf := new(bytes.Buffer)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			fmt.Fprintf(buf, "%10.2f ", m.at(i, j))
		}
		fmt.Fprintln(buf)
	}
	return buf.String()
}

// Rows returns the number of rows in the matrix.
func (m Matrix) Rows() int { return m.rows }

// Cols returns the number of cols in the matrix.
func (m Matrix) Cols() int { return m.cols }

func (m Matrix) checkPos(i, j int) {
	if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
		panic(fmt.Sprintf("element (%d, %d) is out of range", i, j))
	}
}

// At returns the value of the cell at the given position.
// It panics if the position is not valid.
func (m Matrix) At(i, j int) float64 {
	m.checkPos(i, j)
	return m.at(i, j)
}

func (m Matrix) at(i, j int) float64 { return m.data[m.cols*i+j] }

// Clone returns a copy of the current Matrix.
func (m Matrix) Clone() Matrix {
	c := New(m.rows, m.cols)
	copy(c.data, m.data)
	return c
}

func (m Matrix) Apply(f func(i, j int) float64) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.set(i, j, f(i, j))
		}
	}
}

// Set sets the value of the cell at the given position.
// It panics if the position is not valid.
func (m Matrix) Set(i, j int, x float64) Matrix {
	m.checkPos(i, j)
	r := m.Clone()
	r.set(i, j, x)
	return r
}

func (m *Matrix) set(i, j int, x float64) { m.data[m.cols*i+j] = x }

// Scale multiplies the receiver matrix by the given scalar.
func (m Matrix) Scale(x float64) Matrix {
	return Map(func(v float64) float64 { return x * v }, m)
}

// AddScalar adds the given scalar to every cell in the receiver matrix.
func (m Matrix) AddScalar(x float64) Matrix {
	return Map(func(v float64) float64 { return x + v }, m)
}

// Map returns a new Matrix where each value is the result of calling f with the
// value of that position in the original matrix.
func Map(f func(float64) float64, m Matrix) Matrix {
	r := m.Clone()
	for i, v := range r.data {
		r.data[i] = f(v)
	}
	return r
}

// ConcatenateCols returns a matrix that contains the values of all of the given
// matrices side by side.
// All of the matrices need to have the same number of rows.
// The resulting matrix has as many columns as all of the given matrices combined,
// and as many rows as each one of them.
func ConcatenateCols(ms ...Matrix) Matrix {
	if len(ms) == 0 {
		panic("can't concatenate an empty list of matrices")
	}

	rows, cols := ms[0].Rows(), 0
	for _, m := range ms {
		if rows != m.Rows() {
			panic("can't concatenate columns on matrices with different number of rows")
		}
		cols += m.Cols()
	}

	data := make([]float64, 0, rows*cols)
	for i := 0; i < rows; i++ {
		for _, m := range ms {
			data = append(data, m.data[i*m.cols:(i+1)*m.cols]...)
		}
	}

	return Matrix{rows, cols, data}
}

// ConcatenateRows returns a matrix that contains the values of all of the given
// matrices stacked verticaly.
// All of the matrices need to have the same number of columns.
// The resulting matrix has as many rows as all of the given matrices combined,
// and as many cols as each one of them.
func ConcatenateRows(ms ...Matrix) Matrix {
	if len(ms) == 0 {
		panic("can't concatenate an empty list of matrices")
	}

	rows, cols := 0, ms[0].Cols()
	for _, m := range ms {
		if cols != m.Cols() {
			panic("can't concatenate rows on matrices with different number of cols")
		}
		rows += m.Rows()
	}

	data := make([]float64, 0, rows*cols)
	for _, m := range ms {
		data = append(data, m.data...)
	}

	return Matrix{rows, cols, data}
}

// SliceCols returns a new matrix that contains only the columns in between
// from and to, without including to. Similar to slice[from:to].
func (m Matrix) SliceCols(from, to int) Matrix {
	if from < 0 || to > m.cols || to < from {
		panic("bad row numbers")
	}

	data := make([]float64, 0, m.rows*(to-from))
	for i := 0; i < m.rows; i++ {
		data = append(data, m.data[i*m.cols+from:i*m.cols+to]...)
	}
	return Matrix{m.rows, to - from, data}
}

// SliceRows returns a new matrix that contains only the rows in between
// from and to, without including to. Similar to slice[from:to].
func (m Matrix) SliceRows(from, to int) Matrix {
	if from < 0 || to > m.rows || to < from {
		panic("bad row numbers")
	}

	data := make([]float64, m.cols*(to-from))
	copy(data, m.data[from*m.cols:to*m.cols])
	return Matrix{to - from, m.cols, data}
}

// Sum returns the sum of all of the elements in the matrix.
func (m Matrix) Sum() float64 {
	return m.Reduce(0, func(x, cum float64) float64 { return x + cum })
}

// Sum returns the sum of all of the elements in the matrix.
func Sum(m Matrix) float64 { return m.Sum() }

// Reduce provides a functional way of reducing a function over the whole matrix.
// For instance: sum can be implemented as:
//   m.Reduce(0, func(x, cum float64) float64 {return x+cum})
func (m Matrix) Reduce(zero float64, f func(x, cum float64) float64) float64 {
	cum := zero
	for _, x := range m.data {
		cum = f(x, cum)
	}
	return cum
}

// FilterRows returns a matrix where only the rows with an index for which f
// returns true have been kept.
func (m Matrix) FilterRows(f func(i int) bool) Matrix {
	rows, cols := 0, m.Cols()
	var data []float64
	for i := 0; i < m.rows; i++ {
		if f(i) {
			data = append(data, m.data[i*cols:(i+1)*m.cols]...)
			rows++
		}
	}
	return Matrix{rows, cols, data}
}

// T returns the transposed matrix.
func (m Matrix) T() Matrix {
	t := Matrix{m.cols, m.rows, make([]float64, len(m.data))}

	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			t.set(j, i, m.at(i, j))
		}
	}

	return t
}

// Equals returns whether two matrices are identical.
func Equals(a, b Matrix) bool {
	if a.cols != b.cols || a.rows != b.rows {
		return false
	}

	for i, v := range a.data {
		if v != b.data[i] {
			return false
		}
	}
	return true
}

// Product returns the product of two matrices.
func Product(a, b Matrix) Matrix {
	if a.cols != b.rows {
		panicf("can't compute product of matrices with dimensions %dx%d and %dx%d",
			a.rows, a.cols, b.rows, b.cols)
	}

	c := New(a.rows, b.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < b.cols; j++ {
			p := 0.0
			for k := 0; k < a.cols; k++ {
				p += a.at(i, k) * b.at(k, j)
			}
			c.set(i, j, p)
		}
	}
	return c
}

// ParallelProduct returns the product of two matrices performed in parallel.
func ParallelProduct(a, b Matrix) Matrix {
	if a.cols != b.rows {
		panicf("can't compute product of matrices with dimensions %dx%d and %dx%d",
			a.rows, a.cols, b.rows, b.cols)
	}

	var wg sync.WaitGroup
	c := New(a.rows, b.cols)
	for i := 0; i < a.rows; i++ {
		wg.Add(1)
		go func(i int) {
			for j := 0; j < b.cols; j++ {
				p := 0.0
				for k := 0; k < a.cols; k++ {
					p += a.at(i, k) * b.at(k, j)
				}
				c.set(i, j, p)
			}
			wg.Done()
		}(i)
	}
	wg.Wait()
	return c
}

// BlasProduct returns the product of two matrices performed with blas.
func BlasProduct(a, b Matrix) Matrix {
	c := New(a.Rows(), b.Cols())
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0, generalFromMat(a), generalFromMat(b), 0.0, generalFromMat(c))
	return c
}

func generalFromMat(m Matrix) blas64.General {
	return blas64.General{
		Rows:   m.Rows(),
		Cols:   m.Cols(),
		Stride: m.Cols(),
		Data:   m.ToSlice(),
	}
}

// Dot returns the dot product of two matrices.
func Dot(a, b Matrix) Matrix {
	return dotApply(a, b, func(x, y float64) float64 { return x * y })
}

// Plus returns the sum of the two matrices.
func Plus(a, b Matrix) Matrix {
	return dotApply(a, b, func(x, y float64) float64 { return x + y })
}

// Minus returns the difference of the two matrices.
func Minus(a, b Matrix) Matrix {
	return dotApply(a, b, func(x, y float64) float64 { return x - y })
}

func dotApply(a, b Matrix, f func(x, y float64) float64) Matrix {
	if a.cols != b.cols || a.rows != b.rows {
		panicf("can't compute dot application of matrices with dimensions %dx%d and %dx%d",
			a.rows, a.cols, b.rows, b.cols)
	}

	c := New(a.rows, a.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < b.cols; j++ {
			c.set(i, j, f(a.at(i, j), b.at(i, j)))
		}
	}
	return c
}

func panicf(format string, args ...interface{}) {
	panic(fmt.Sprintf(format, args...))
}
