#include "bounds.h"

#include "test.h"

namespace nda {

TEST(bounds_0d) {
  static_assert(bounds<>::is_scalar());
  static_assert(bounds<>::rank() == 0);

  bounds<> b0;
  bounds<> b1 = make_bounds();

  bounds<> b2 = b0;
  bounds<> b3(b1);

  bounds<> b4 = std::move(b3);
  (void)b4;
}

TEST(bounds_1d_static) {
  // Fully static.
  bounds<interval<3, 4>> b0;
  ASSERT(b0.interval<0>().min() == 3);
  ASSERT(b0.interval<0>().extent() == 4);
  // TODO(jiawen): This is a cumbersome way to manipulate a rectangle for many
  // use cases. Add a dynamic accessor?
  ASSERT(std::get<0>(b0.min()) == 3);
  ASSERT(std::get<0>(b0.extent()) == 4);
  ASSERT(std::get<0>(b0.max()) == 6);

  // Partially static.
  bounds<interval<7>> b1;
  b1.interval<0>().set_extent(8);
  ASSERT(b1.interval<0>().min() == 7);
  ASSERT(b1.interval<0>().extent() == 8);
  ASSERT(std::get<0>(b1.min()) == 7);
  ASSERT(std::get<0>(b1.extent()) == 8);
  ASSERT(std::get<0>(b1.max()) == 14);

  bounds<interval<dynamic, 10>> b2;
  b2.interval<0>().set_min(11);
  ASSERT(b2.interval<0>().min() == 11);
  ASSERT(b2.interval<0>().extent() == 10);
  ASSERT(std::get<0>(b2.min()) == 11);
  ASSERT(std::get<0>(b2.extent()) == 10);
  ASSERT(std::get<0>(b2.max()) == 20);

  // Fully dynamic.
  bounds<interval<>> b3;
  ASSERT(b3.interval<0>().min() == 0);
  ASSERT(b3.interval<0>().extent() == 1);
  b3.interval<0>().set_min(7);
  b3.interval<0>().set_extent(8);
  ASSERT(b3.interval<0>().min() == 7);
  ASSERT(b3.interval<0>().extent() == 8);
  ASSERT(std::get<0>(b3.min()) == 7);
  ASSERT(std::get<0>(b3.extent()) == 8);
  ASSERT(std::get<0>(b3.max()) == 14);
}

TEST(bounds_2d) {
  // Also checks bounds::dims(), which creates a std::tuple<nda::dim<min, extent>...>.

  // Fully static.
  bounds<interval<0, 4>, interval<1, 6>> b0;
  ASSERT(b0.min() == std::make_tuple<index_t>(0, 1));
  ASSERT(b0.extent() == std::make_tuple<index_t>(4, 6));

  std::tuple<dim<0, 4>, dim<1, 6>> d0 = b0.dims();
  ASSERT(std::get<0>(d0).min() == 0);
  ASSERT(std::get<1>(d0).min() == 1);
  ASSERT(std::get<0>(d0).extent() == 4);
  ASSERT(std::get<1>(d0).extent() == 6);

  // Partially static.
  bounds<interval<dynamic, 100>, interval<200, dynamic>> b1;
  b1.interval<0>().set_min(9);
  b1.interval<1>().set_extent(11);
  ASSERT(b1.min() == std::make_tuple<index_t>(9, 200));
  ASSERT(b1.extent() == std::make_tuple<index_t>(100, 11));

  std::tuple<dim<>, dim<>> d1 = b1.dims();
  ASSERT(std::get<0>(d1).min() == 9);
  ASSERT(std::get<1>(d1).min() == 200);
  ASSERT(std::get<0>(d1).extent() == 100);
  ASSERT(std::get<1>(d1).extent() == 11);

  // Fully dynamic.
  bounds<interval<>, interval<>> b2(r(0, 3), r(2, 8));
  ASSERT(b2.min() == std::make_tuple<index_t>(0, 2));
  ASSERT(b2.extent() == std::make_tuple<index_t>(3, 6));

  std::tuple<dim<>, dim<>> d2 = b2.dims();
  ASSERT(std::get<0>(d2).min() == 0);
  ASSERT(std::get<1>(d2).min() == 2);
  ASSERT(std::get<0>(d2).extent() == 3);
  ASSERT(std::get<1>(d2).extent() == 6);
}

TEST(bounds_of_rank) {
  // 0D.
  {
    bounds_of_rank<0> b;
    ASSERT(b.is_scalar());
    ASSERT(b.rank() == 0);
  }

  // 2D.
  {
    // Dynamic constructor.
    bounds_of_rank<2> b0(r(3, 4), r(5, 6));

    // Dynamic from two fixed.
    bounds_of_rank<2> b1(r<4>(3), r<6>(5));

    // Dynamic from one fixed.
    bounds_of_rank<2> b2(r<4>(3), r(5, 6));
    bounds_of_rank<2> b3(r(3, 4), r<6>(5));
  }

  // 4D.
  {
    bounds_of_rank<4> b;
    b.interval<0>().set_extent(6);
    b.interval<1>().set_extent(4);
    b.interval<2>().set_min(2);
    b.interval<3>().set_min(-5);
    ASSERT(b.min() == bounds_of_rank<4>::index_type(0, 0, 2, -5));
    ASSERT(b.extent() == bounds_of_rank<4>::index_type(6, 4, 1, 1));
    ASSERT(b.max() == bounds_of_rank<4>::index_type(5, 3, 2, -5));
  }
}

// TODO(jiawen): @dsharlet: it'd be nice to have some more interval conveniences:
// - a type alias for an interval with a fixed min that defaults to 0. min_interval?
//   - But this isn't any different than interval<Min, Extent> (besides the zero default).
//   So may zero_interval is a better name?
// - an alias to make an interval with min = 0. v(extent) and v<extent>() like we talked about?
template <index_t Min = 0>
using min_interval = interval<Min, dynamic>;

TEST(shape_from_bounds) {
  // Scalar.
  {
    auto sh = make_shape(bounds<>());
    static_assert(decltype(sh)::is_scalar());
    static_assert(std::is_same<decltype(sh), shape<>>::value);
  }

  // shape_of_rank<r> from bounds_of_rank<r>.
  {
    bounds_of_rank<2> b;
    b.interval<0>().set_extent(6);
    b.interval<1>().set_min(100);
    b.interval<1>().set_extent(4);

    shape_of_rank<2> sh(b);
    ASSERT(sh.dim<0>().min() == 0);
    ASSERT(sh.dim<0>().extent() == 6);
    ASSERT(sh.dim<1>().min() == 100);
    ASSERT(sh.dim<1>().extent() == 4);
  }

  // fixed_intervals map to dimensions with fixed extents
  {
    bounds<fixed_interval<640>, fixed_interval<480>, fixed_interval<3>> b;

    shape_of_rank<3> sh0(b);
    ASSERT(sh0.dim<0>().min() == 0);
    ASSERT(sh0.dim<0>().extent() == 640);
    ASSERT(sh0.dim<1>().min() == 0);
    ASSERT(sh0.dim<1>().extent() == 480);
    ASSERT(sh0.dim<2>().min() == 0);
    ASSERT(sh0.dim<2>().extent() == 3);

    // This should fail to build:
    // shape_of_rank<2> shape_of_wrong_rank(b);

    b.interval<0>().set_min(5);
    b.interval<1>().set_min(10);

    // ELEPHANT
    // @dsharlet: this doesn't crash with an assert - it just has no effect
    // fixed_interval::set_extent() doesn't check anything
    // b.interval<1>().set_extent(10);

    shape<dim<dynamic, 640>, dim<dynamic, 480>, dim<dynamic, 3>> sh1 = make_shape(b);
    ASSERT(sh1.dim<0>().min() == 5);
    ASSERT(sh1.dim<0>().extent() == 640);
    ASSERT(sh1.dim<1>().min() == 10);
    ASSERT(sh1.dim<1>().extent() == 480);
    ASSERT(sh1.dim<2>().min() == 0);
    ASSERT(sh1.dim<2>().extent() == 3);

    // This should fail to build:
    // shape<dim<dynamic, 641>, dim<dynamic, 480>, dim<dynamic, 3>> sh1(b);
  }

  // min_intervals map to dimensions with fixed mins.
  {
    bounds<min_interval<17>, min_interval<>, fixed_interval<3>, interval<>> b;
    shape_of_rank<4> sh0(b);
    ASSERT(sh0.dim<0>().min() == 17);
    ASSERT(sh0.dim<0>().extent() == 1);
    ASSERT(sh0.dim<1>().min() == 0);
    ASSERT(sh0.dim<1>().extent() == 1);
    ASSERT(sh0.dim<2>().min() == 0);
    ASSERT(sh0.dim<2>().extent() == 3);
    ASSERT(sh0.dim<3>().min() == 0);
    ASSERT(sh0.dim<3>().extent() == 1);

    b.interval<0>().set_extent(1024);
    b.interval<1>().set_extent(768);
    b.interval<2>().set_min(-7);
    b.interval<3>().set_min(-12);
    b.interval<3>().set_extent(4);
    shape<dim<17>, dim<0>, dim<dynamic, 3>, dim<>> sh1(b);
    ASSERT(sh1.dim<0>().min() == 17);
    ASSERT(sh1.dim<0>().extent() == 1024);
    ASSERT(sh1.dim<1>().min() == 0);
    ASSERT(sh1.dim<1>().extent() == 768);
    ASSERT(sh1.dim<2>().min() == -7);
    ASSERT(sh1.dim<2>().extent() == 3);
    ASSERT(sh1.dim<3>().min() == -12);
    ASSERT(sh1.dim<3>().extent() == 4);

    shape<dim<17>, dim<0>, dim<dynamic, 3>, dim<>> sh2 = make_shape(b);
    ASSERT(sh2.dim<0>().min() == 17);
    ASSERT(sh2.dim<0>().extent() == 1024);
    ASSERT(sh2.dim<1>().min() == 0);
    ASSERT(sh2.dim<1>().extent() == 768);
    ASSERT(sh2.dim<2>().min() == -7);
    ASSERT(sh2.dim<2>().extent() == 3);
    ASSERT(sh2.dim<3>().min() == -12);
    ASSERT(sh2.dim<3>().extent() == 4);

    // Fully dynamic works too.
    shape_of_rank<decltype(b)::rank()> sh3(b);
    ASSERT(sh3.dim<0>().min() == 17);
    ASSERT(sh3.dim<0>().extent() == 1024);
    ASSERT(sh3.dim<1>().min() == 0);
    ASSERT(sh3.dim<1>().extent() == 768);
    ASSERT(sh3.dim<2>().min() == -7);
    ASSERT(sh3.dim<2>().extent() == 3);
    ASSERT(sh3.dim<3>().min() == -12);
    ASSERT(sh3.dim<3>().extent() == 4);

    // This should fail to build:
    // shape<dim<18>, dim<0>, dim<dynamic, 3>, dim<>> sh1(b);
  }
}

} // namespace nda
