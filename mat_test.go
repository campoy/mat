package mat

import (
	"fmt"
	"math"
	"testing"
)

func TestNewMatrix(t *testing.T) {
	m := New(2, 3)
	fmt.Println(m)
	m = m.AddScalar(1)
	fmt.Println(m)
	m = m.Scale(math.Pi)
	fmt.Println(m)
}

func TestConcatenateCols(t *testing.T) {
	ones := New(3, 1).AddScalar(1)
	twos := New(3, 2).AddScalar(2)
	threes := New(3, 3).AddScalar(3)

	fmt.Println(ConcatenateCols(ones, twos, threes))
}

func TestConcatenateRows(t *testing.T) {
	ones := New(1, 3).AddScalar(1)
	twos := New(2, 3).AddScalar(2)
	threes := New(3, 3).AddScalar(3)

	fmt.Println(ConcatenateRows(ones, twos, threes))
}

func TestSliceCols(t *testing.T) {
	m := ConcatenateCols(
		New(3, 1).AddScalar(1),
		New(3, 2).AddScalar(2),
		New(3, 3).AddScalar(3))

	fmt.Println(m.SliceCols(0, 1))
	fmt.Println(m.SliceCols(1, 3))
	fmt.Println(m.SliceCols(3, 6))
}

func TestSliceRows(t *testing.T) {
	m := ConcatenateRows(
		New(1, 3).AddScalar(1),
		New(2, 3).AddScalar(2),
		New(3, 3).AddScalar(3))

	fmt.Println(m)
	fmt.Println(m.SliceRows(0, 1))
	fmt.Println(m.SliceRows(1, 3))
	fmt.Println(m.SliceRows(3, 6))
}

func TestFilterRows(t *testing.T) {
	m := ConcatenateRows(
		New(1, 3).AddScalar(1),
		New(1, 3).AddScalar(2),
		New(1, 3).AddScalar(3),
		New(1, 3).AddScalar(4),
		New(1, 3).AddScalar(5),
		New(1, 3).AddScalar(6))

	res := m.FilterRows(func(i int) bool { return i%2 == 1 })
	fmt.Printf("After filtering we have %d rows\n", res.Rows())
	fmt.Println(res)
}

func TestTranspose(t *testing.T) {
	m := ConcatenateRows(
		New(1, 3).AddScalar(1),
		New(2, 3).AddScalar(2),
		New(3, 3).AddScalar(3))

	fmt.Println(m.T())
}

func TestProduct(t *testing.T) {
	tt := []struct {
		a, b, c Matrix
	}{
		{New(2, 3).AddScalar(1), New(3, 2).AddScalar(2), New(2, 2).AddScalar(6)},
		{New(1, 10).AddScalar(1), New(10, 1).AddScalar(1), New(1, 1).AddScalar(10)},
		{New(10, 1).AddScalar(1), New(1, 10).AddScalar(1), New(10, 10).AddScalar(1)},
	}

	for _, tc := range tt {
		if p := Product(tc.a, tc.b); !Equals(p, tc.c) {
			t.Errorf("expected result:\n%v\ngot:\n%v\n", tc.c, p)
		}
	}
}

func TestProductGPU(t *testing.T) {
	tt := []struct {
		a, b, c Matrix
	}{
		{New(2, 3).AddScalar(1), New(3, 2).AddScalar(2), New(2, 2).AddScalar(6)},
		{New(1, 10).AddScalar(1), New(10, 1).AddScalar(1), New(1, 1).AddScalar(10)},
		{New(10, 1).AddScalar(1), New(1, 10).AddScalar(1), New(10, 10).AddScalar(1)},
	}

	for _, tc := range tt {
		if p := Product(tc.a, tc.b); !Equals(p, tc.c) {
			t.Errorf("expected result:\n%v\ngot:\n%v\n", tc.c, p)
		}
	}
}
