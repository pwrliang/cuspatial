/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/bounding_boxes.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/geometry/box.hpp>
#include <cuspatial/geometry/vec_2d.hpp>

#include <gtest/gtest.h>

template <typename T>
struct MultiPolygonBoundingBoxTest : public ::testing::Test {};

using cuspatial::vec_2d;
using cuspatial::test::make_device_vector;

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(MultiPolygonBoundingBoxTest, TestTypes);

TYPED_TEST(MultiPolygonBoundingBoxTest, test_empty)
{
  using T = TypeParam;

  {
    auto geom_offsets = make_device_vector<int32_t>({});
    auto part_offsets = make_device_vector<int32_t>({});
    auto ring_offsets = make_device_vector<int32_t>({});
    auto vertices     = make_device_vector<vec_2d<T>>({});

    auto bboxes = rmm::device_vector<cuspatial::box<T>>(0);

    auto bboxes_end = cuspatial::polygon_bounding_boxes(geom_offsets.begin(),
                                                        geom_offsets.end(),
                                                        part_offsets.begin(),
                                                        part_offsets.end(),
                                                        ring_offsets.begin(),
                                                        ring_offsets.end(),
                                                        vertices.begin(),
                                                        vertices.end(),
                                                        bboxes.begin());

    EXPECT_EQ(std::distance(bboxes.begin(), bboxes_end), 0);
  }

  {
    auto geom_offsets = make_device_vector<int32_t>({0});
    auto part_offsets = make_device_vector<int32_t>({});
    auto ring_offsets = make_device_vector<int32_t>({});
    auto vertices     = make_device_vector<vec_2d<T>>({});

    auto bboxes = rmm::device_vector<cuspatial::box<T>>(0);

    auto bboxes_end = cuspatial::polygon_bounding_boxes(geom_offsets.begin(),
                                                        geom_offsets.end(),
                                                        part_offsets.begin(),
                                                        part_offsets.end(),
                                                        ring_offsets.begin(),
                                                        ring_offsets.end(),
                                                        vertices.begin(),
                                                        vertices.end(),
                                                        bboxes.begin());

    EXPECT_EQ(std::distance(bboxes.begin(), bboxes_end), 0);
  }
}

TYPED_TEST(MultiPolygonBoundingBoxTest, test_small)
{
  using T = TypeParam;

  // GeoArrow: Final offset points to the end of the data. The number of offsets is number of
  // geometries / parts plus one.
  auto geom_offsets = make_device_vector<int32_t>({0, 3, 4});
  auto part_offsets = make_device_vector<int32_t>({0, 1, 1, 2, 3});
  auto ring_offsets = make_device_vector<int32_t>({0, 4, 8, 12});
  auto vertices     = make_device_vector<vec_2d<T>>({{0, 0},
                                                     {1, 0},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0.2, 0.2},
                                                     {0.2, 0.3},
                                                     {0.3, 0.3},
                                                     {0.2, 0.2},
                                                     {0, 0},
                                                     {1, 0},
                                                     {1, 1},
                                                     {0, 0}});

  // GeoArrow: Number of linestrings is number of offsets minus one.
  auto bboxes     = rmm::device_vector<cuspatial::box<T>>(geom_offsets.size() - 1);
  auto bboxes_end = cuspatial::polygon_bounding_boxes(geom_offsets.begin(),
                                                      geom_offsets.end(),
                                                      part_offsets.begin(),
                                                      part_offsets.end(),
                                                      ring_offsets.begin(),
                                                      ring_offsets.end(),
                                                      vertices.begin(),
                                                      vertices.end(),
                                                      bboxes.begin());
  EXPECT_EQ(std::distance(bboxes.begin(), bboxes_end), 2);

  auto bboxes_expected = make_device_vector<cuspatial::box<T>>({
    {{0, 0}, {1, 1}},  // geom 1
    {{0, 0}, {1, 1}}   // geom 2
  });

  CUSPATIAL_EXPECT_VEC2D_PAIRS_EQUIVALENT(bboxes, bboxes_expected);
}
