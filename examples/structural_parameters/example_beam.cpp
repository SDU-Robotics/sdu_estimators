#include <fstream>
#include <iostream>
#include <iomanip>

#include <sdu_estimators/parameter_estimators/gradient_estimator.hpp>
#include "sdu_estimators/regressor_extensions/kreisselmeier.hpp"
#include <sdu_estimators/parameter_estimators/drem.hpp>


Eigen::Matrix<long double, 182, 13> getBeamPhi(Eigen::Vector<long double, 13> x, Eigen::Vector<long double, 13> ddx)
{
  /* Conversion from MATLAB code to C++ for phi-matrix for the EUROfusion beam.
   *
   * ddx1 = in2(1,:);
  ddx2 = in2(2,:);
  ddx3 = in2(3,:);
  ddx4 = in2(4,:);
  ddx5 = in2(5,:);
  ddx6 = in2(6,:);
  ddx7 = in2(7,:);
  ddx8 = in2(8,:);
  ddx9 = in2(9,:);
  ddx10 = in2(10,:);
  ddx11 = in2(11,:);
  ddx12 = in2(12,:);
  ddx13 = in2(13,:);
  x1 = in1(1,:);
  x2 = in1(2,:);
  x3 = in1(3,:);
  x4 = in1(4,:);
  x5 = in1(5,:);
  x6 = in1(6,:);
  x7 = in1(7,:);
  x8 = in1(8,:);
  x9 = in1(9,:);
  x10 = in1(10,:);
  x11 = in1(11,:);
  x12 = in1(12,:);
  x13 = in1(13,:);
  mt1 = [ddx1,ddx2,ddx3,ddx4,ddx5,ddx6,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,ddx3,ddx4,ddx5,ddx6];
  mt2 = [ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,ddx4,ddx5,ddx6,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13];
  mt3 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,ddx5,ddx6,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
  mt4 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,ddx6,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
  mt5 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
  mt6 = [0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
  mt7 = [0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,0.0,0.0,0.0,0.0,0.0,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0];
  mt8 = [0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,0.0,0.0,0.0,0.0,0.0,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,0.0,0.0,0.0,0.0,0.0,ddx8,0.0,0.0,0.0,0.0,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0];
  mt9 = [0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,0.0,0.0,0.0,0.0,0.0,x8,0.0,0.0,0.0,0.0,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,0.0,0.0,0.0,0.0,0.0,ddx8,0.0,0.0,0.0,0.0,ddx9,0.0,0.0,0.0,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,0.0,0.0,0.0,0.0];
  mt10 = [0.0,x8,0.0,0.0,0.0,0.0,x9,0.0,0.0,0.0,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,0.0,0.0,0.0,0.0,0.0,ddx8,0.0,0.0,0.0,0.0,ddx9,0.0,0.0,0.0,ddx10,0.0,0.0,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,0.0,0.0,0.0,0.0,0.0,x8,0.0,0.0,0.0,0.0,x9,0.0,0.0,0.0,x10,0.0,0.0,x11,x12,x13,0.0];
  mt11 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,0.0,0.0,0.0,0.0,0.0,ddx8,0.0,0.0,0.0,0.0,ddx9,0.0,0.0,0.0,ddx10,0.0,0.0,ddx11,0.0,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,0.0,0.0,0.0,0.0,0.0,x8,0.0,0.0,0.0,0.0,x9,0.0,0.0,0.0,x10,0.0,0.0,x11,0.0,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0];
  mt12 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,0.0,0.0,0.0,0.0,0.0,ddx8,0.0,0.0,0.0,0.0,ddx9,0.0,0.0,0.0,ddx10,0.0,0.0,ddx11,0.0,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,0.0,0.0,0.0,0.0,0.0,x8,0.0,0.0,0.0,0.0,x9,0.0,0.0,0.0,x10,0.0,0.0,x11,0.0,x12,x13];
  out1 = reshape([mt1,mt2,mt3,mt4,mt5,mt6,mt7,mt8,mt9,mt10,mt11,mt12],182,13);
  */

  Eigen::Vector<long double, 200> mt1, mt2, mt3, mt4, mt5, mt6, mt7, mt8, mt9, mt10, mt11;
  Eigen::Vector<long double, 166> mt12;

  // mt1 = [ddx1,ddx2,ddx3,ddx4,ddx5,ddx6,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,ddx3,ddx4,ddx5,ddx6];
  long double ddx1, ddx2, ddx3, ddx4, ddx5, ddx6, ddx7, ddx8, ddx9, ddx10, ddx11, ddx12, ddx13;
  ddx1 = ddx[0]; ddx2 = ddx[1]; ddx3 = ddx[2]; ddx4 = ddx[3]; ddx5 = ddx[4]; ddx6 = ddx[5]; ddx7 = ddx[6]; ddx8 = ddx[7];
  ddx9 = ddx[8]; ddx10 = ddx[9]; ddx11 = ddx[10]; ddx12 = ddx[11]; ddx13 = ddx[12];

  long double x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13;
  x1 = x[0]; x2 = x[1]; x3 = x[2]; x4 = x[3]; x5 = x[4]; x6 = x[5]; x7 = x[6]; x8 = x[7];
  x9 = x[8]; x10 = x[9]; x11 = x[10]; x12 = x[11]; x13 = x[12];

  mt1 << ddx1,ddx2,ddx3,ddx4,ddx5,ddx6,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,ddx3,ddx4,ddx5,ddx6;
  mt2 << ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,ddx4,ddx5,ddx6,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13;
  mt3 << 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,ddx5,ddx6,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0;
  mt4 << 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,ddx6,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0;
  mt5 << 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0;
  mt6 << 0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0;
  mt7 << 0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,0.0,0.0,0.0,0.0,0.0,ddx8,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0;
  mt8 << 0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,0.0,0.0,0.0,0.0,0.0,x8,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,0.0,0.0,0.0,0.0,0.0,ddx8,0.0,0.0,0.0,0.0,ddx9,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0;
  mt9 << 0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,0.0,0.0,0.0,0.0,0.0,x8,0.0,0.0,0.0,0.0,x9,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,0.0,0.0,0.0,0.0,0.0,ddx8,0.0,0.0,0.0,0.0,ddx9,0.0,0.0,0.0,ddx10,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,0.0,0.0,0.0,0.0;
  mt10 << 0.0,x8,0.0,0.0,0.0,0.0,x9,0.0,0.0,0.0,x10,x11,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,0.0,0.0,0.0,0.0,0.0,ddx8,0.0,0.0,0.0,0.0,ddx9,0.0,0.0,0.0,ddx10,0.0,0.0,ddx11,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,0.0,0.0,0.0,0.0,0.0,x8,0.0,0.0,0.0,0.0,x9,0.0,0.0,0.0,x10,0.0,0.0,x11,x12,x13,0.0;
  mt11 << 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,0.0,0.0,0.0,0.0,0.0,ddx8,0.0,0.0,0.0,0.0,ddx9,0.0,0.0,0.0,ddx10,0.0,0.0,ddx11,0.0,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,0.0,0.0,0.0,0.0,0.0,x8,0.0,0.0,0.0,0.0,x9,0.0,0.0,0.0,x10,0.0,0.0,x11,0.0,x12,x13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx1,0.0,0.0,0.0;
  mt12 << 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,ddx6,0.0,0.0,0.0,0.0,0.0,0.0,ddx7,0.0,0.0,0.0,0.0,0.0,ddx8,0.0,0.0,0.0,0.0,ddx9,0.0,0.0,0.0,ddx10,0.0,0.0,ddx11,0.0,ddx12,ddx13,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,x6,0.0,0.0,0.0,0.0,0.0,0.0,x7,0.0,0.0,0.0,0.0,0.0,x8,0.0,0.0,0.0,0.0,x9,0.0,0.0,0.0,x10,0.0,0.0,x11,0.0,x12,x13;

  // Eigen::Matrix<long double, 182, 13> out;
  Eigen::Vector<long double, 2366> out_vec;
  out_vec << mt1, mt2, mt3, mt4, mt5, mt6, mt7, mt8, mt9, mt10, mt11, mt12;

  // Reshape to 182x13
  Eigen::Matrix<long double, 182, 13> out_mat;
  out_mat << out_vec.reshaped(182, 13);

  return out_mat;
}

Eigen::Matrix<long double, 13, 13> getTrueM()
{
  Eigen::Matrix<long double, 13, 13> out;
  out <<
     25644.5988898207,19.3338614509,-6411.4751410036,570.4311625169,-508.1774146847,6765.5236447759,4153.5934224066,705.3490395516,4786.8390813827,-8810.6227682793,-40768.1831291458,17392.1165380621,26.8229032913,
     19.3338614509,27841.7726022066,5335.8092827834,2851.4088719601,-544.7876326571,-3194.4068860441,-986.7418003712,6206.7839780927,-3138.8342965132,-7027.4111544981,30852.4864384382,65714.2057504058,27.5170217960,
     -6411.4751410036,5335.8092827834,61881.0371750508,-978.0583712796,3764.2001298737,39.4219068852,-18812.9151901481,1042.0445194096,-18490.3222427406,24909.0055674542,155761.4520473060,-17021.9117425863,-14.9301053825,
     570.4311625169,2851.4088719601,-978.0583712796,1611.3629836892,5.1141804844,5141.4192323827,871.9485398128,3885.1554159143,-617.7698235038,-6803.8577289873,8519.7491222095,46635.2604813342,37.4657026809,
     -508.1774146847,-544.7876326571,3764.2001298737,5.1141804844,995.9033519039,549.8037723648,-1566.3073795034,-294.7600580355,-3282.9377504476,4921.8223369311,30325.0880931678,-5167.1931971961,-0.7660824068,
     6765.5236447759,-3194.4068860441,39.4219068852,5141.4192323827,549.8037723648,29767.3968649924,2585.7550733027,11654.4640542061,-758.2082510226,-23916.1350005012,15724.1645972879,148536.0259438270,158.2641522256,
     4153.5934224066,-986.7418003712,-18812.9151901481,871.9485398128,-1566.3073795034,2585.7550733027,26087.6653095597,262.0588993728,20437.5512497871,-29328.3081477976,-170574.4344862930,41182.4230791310,22.1159369099,
     705.3490395516,6206.7839780927,1042.0445194096,3885.1554159143,-294.7600580355,11654.4640542061,262.0588993728,19784.1104859212,-3239.0195056660,-22988.2302737082,35269.2106977169,209929.7274794590,94.7321294457,
     4786.8390813827,-3138.8342965132,-18490.3222427406,-617.7698235038,-3282.9377504476,-758.2082510226,20437.5512497871,-3239.0195056660,35139.0583546388,-42422.3537139804,-316685.9456916000,-5727.1902143511,-4.8434036237,
     -8810.6227682793,-7027.4111544981,24909.0055674542,-6803.8577289873,4921.8223369311,-23916.1350005012,-29328.3081477976,-22988.2302737082,-42422.3537139804,96992.0211753947,369902.2956451150,-298025.0231570660,-181.4834292607,
     -40768.1831291458,30852.4864384382,155761.4520473060,8519.7491222095,30325.0880931678,15724.1645972879,-170574.4344862930,35269.2106977169,-316685.9456916000,369902.2956451150,2872124.6227729502,141894.0749190570,108.8506047925,
     17392.1165380621,65714.2057504058,-17021.9117425863,46635.2604813342,-5167.1931971961,148536.0259438270,41182.4230791310,209929.7274794590,-5727.1902143511,-298025.0231570660,141894.0749190570,2313232.3688817401,1155.1615631776,
     26.8229032913,27.5170217960,-14.9301053825,37.4657026809,-0.7660824068,158.2641522256,22.1159369099,94.7321294457,-4.8434036237,-181.4834292607,108.8506047925,1155.1615631776,1.0000000000;

  return out;
}

Eigen::Matrix<long double, 13, 13> getTrueK()
{
  Eigen::Matrix<long double, 13, 13> out;
  out <<
     1407769.8107290301,86337.4898350760,3284758.7589110299,-672481.4604448490,-5301865.1981767099,503078.1037384460,-1407769.8107229001,-86337.4898424598,-3284758.7589101898,5026257.7074759603,27834484.8395485990,-2961256.9762905799,0.0000000000,
     86337.4898350760,789973.9600906370,649353.3509640290,1792509.4631268301,8080.3575629890,371486.1657960120,-86337.4898455659,-789973.9600935800,-649353.3509668360,7353058.8462429298,6503965.0309620304,-9509733.7564682700,0.0000000000,
     3284758.7589110299,649353.3509640290,21914622.4305992015,-2483131.2416875600,-10260870.7354157995,811998.6478523170,-3284758.7589758099,-649353.3509800100,-21914622.4305287004,32315997.6350428015,226054951.3286240101,-11677820.2836585995,0.0000000000,
     -672481.4604448490,1792509.4631268301,-2483131.2416875600,15878213.6263561007,3357508.6016468899,-98336.7560167635,672481.4604473270,-1792509.4631330101,2483131.2416841299,744282.1782301080,-24590168.6477624997,-19730685.3627181984,0.0000000000,
     -5301865.1981767099,8080.3575629890,-10260870.7354157995,3357508.6016468899,33992897.5485692024,-1746634.3997361499,5301865.1981689502,-8080.3575647363,10260870.7354167998,-13980197.6701737009,-94686590.6883516014,7187657.8937981399,0.0000000000,
     503078.1037384460,371486.1657960120,811998.6478523170,-98336.7560167635,-1746634.3997361499,13706579.4291452002,-503078.1037337300,-371486.1658434610,-811998.6478580410,4927832.0477855196,5654474.5493679401,-18486540.4518472999,0.0000000000,
     -1407769.8107229001,-86337.4898455659,-3284758.7589758099,672481.4604473270,5301865.1981689502,-503078.1037337300,1407769.8107910200,86337.4898520708,3284758.7589921998,-5026257.7075843802,-27834484.8401032016,2961256.9764804798,0.0000000000,
     -86337.4898424598,-789973.9600935800,-649353.3509800100,-1792509.4631330101,-8080.3575647363,-371486.1658434610,86337.4898520708,789973.9600715640,649353.3509736060,-7353058.8462212104,-6503965.0310134897,9509733.7562560998,0.0000000000,
     -3284758.7589101898,-649353.3509668360,-21914622.4305287004,2483131.2416841299,10260870.7354167998,-811998.6478580410,3284758.7589921998,649353.3509736060,21914622.4304962009,-32315997.6350403018,-226054951.3283079863,11677820.2835664004,0.0000000000,
     5026257.7074759603,7353058.8462429298,32315997.6350428015,744282.1782301080,-13980197.6701737009,4927832.0477855196,-5026257.7075843802,-7353058.8462212104,-32315997.6350403018,111802445.1872559935,330241309.8676149845,-94393596.2580879033,0.0000000000,
     27834484.8395485990,6503965.0309620304,226054951.3286240101,-24590168.6477624997,-94686590.6883516014,5654474.5493679401,-27834484.8401032016,-6503965.0310134897,-226054951.3283079863,330241309.8676149845,2385494105.8125000000,-109199979.2606039941,0.0000000000,
     -2961256.9762905799,-9509733.7564682700,-11677820.2836585995,-19730685.3627181984,7187657.8937981399,-18486540.4518472999,2961256.9764804798,9509733.7562560998,11677820.2835664004,-94393596.2580879033,-109199979.2606039941,130498997.3522949964,0.0000000000,
     0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,0.0000000000,478.4288048831;

  return out;
}

int main()
{
  float fs = 1000;
  float dt = 1. / fs;
  float tend = 20. / dt;

  Eigen::Vector<long double, 182> theta_init;
  theta_init.setZero();

  Eigen::Vector<long double, 182> gamma;
  gamma.setOnes();

  gamma *= 1e6;
  float r = 1;
  sdu_estimators::parameter_estimators::GradientEstimator<long double, 13, 182> estimator(dt, gamma, theta_init, r);


  /*
  float ell = 0.001;
  sdu_estimators::regressor_extensions::Kreisselmeier<long double, 13, 182> reg_ext(dt, ell);
  float r = 0.;
  gamma *= 1e100;
  sdu_estimators::parameter_estimators::DREM<long double, 13, 182> estimator(dt, gamma, theta_init, &reg_ext, r);
  */

  Eigen::Matrix<long double, 182, 13> phi;
  Eigen::Vector<long double, 13> q, dq, ddq, y, F, u;
  u.setZero();
  F.setZero();

  Eigen::Matrix<long double, 13, 13> M_act, K_act, M_act_inv;

  M_act = getTrueM();
  M_act_inv = M_act.inverse();
  K_act = getTrueK();

  Eigen::Vector<long double, 91> M_act_vec, K_act_vec;
  Eigen::Vector<long double, 182> theta_act, theta_est;

  int id_elem = 0;

  for (int i = 0; i < M_act.cols(); ++i)
  {
    for (int j = i; j < M_act.rows(); ++j)
    {
      M_act_vec[id_elem] = M_act(i, j);
      K_act_vec[id_elem] = K_act(i, j);

      ++id_elem;
    }
  }

  theta_act << M_act_vec, K_act_vec;

  // q.setOnes();
  // q *= 0.5;
  q.setZero();
  // q[0] = 1.;
  // q[1] = 1.;
  // q[2] = 1.;

  dq.setZero();
  ddq.setZero();
  std::vector<Eigen::Vector<long double, 13>> all_q, all_dq, all_ddq, all_F;

  phi = getBeamPhi(q, ddq);

  // std::cout << phi << std::endl;

  std::vector<Eigen::Vector<long double, 182>> all_theta_est;

  std::vector<long double> parameter_error;
  std::vector<long double> residual_error;

  // std::vector<Eigen::VectorXd> all_y;
  // std::vector<Eigen::MatrixXd> all_phi;

  float t;
  for (int i = 0; i < tend; ++i)
  {
    t = i * dt;

    std::cout << t << std::endl;

    // // movement, probably not realistic at all
    // ddq.setRandom();
    // dq += dt * ddq;
    // q += dt * dq;
    //
    // // load true matrices
    // y << K_act * q + M_act * ddq;

    // u.setRandom();
    // u *= 0.001;
    u[0] = 10 * sin(10 * t);
    u[1] = 10 * sin(15 * t + 0.1);
    u[2] = 10 * sin(20 * t + 0.2);

    // std::cout << F << std::endl;

    // ddq = M_act_inv * (F + u - K_act * q);
    // ddq = M_act_inv * (F + u - K_act * q);
    ddq = M_act_inv * (u - K_act * q);
    // ddq = M_act_inv * (F - K_act * q);
    dq += dt * ddq;
    q += dt * dq;

    // std::cout << u << std::endl;
    // ddq = M_act_inv * (u - K_act * q);

    // std::cout << "ddq " << ddq << std::endl;
    // std::cout << "dq " << dq << std::endl;
    // std::cout << "q " << q << std::endl;

    F = M_act * ddq + K_act * q; // - u;
    // F = M_act * ddq + K_act * q;
    // y << -(M_act * ddq + K_act * q);
    y << F;

    // std::cout << y << std::endl;
    phi = getBeamPhi(q, ddq);

    estimator.step(y, phi);

    all_theta_est.push_back(estimator.get_estimate());

    theta_est = estimator.get_estimate();

    // all_y.push_back(y);
    // all_phi.push_back(phi);

    all_q.push_back(q);
    all_dq.push_back(dq);
    all_ddq.push_back(ddq);
    all_F.push_back(F);

    parameter_error.push_back(
      (theta_est - theta_act).squaredNorm()
    );

    residual_error.push_back(
      (y - phi.transpose() * theta_est).squaredNorm()
    );

    // std::cout << theta_est.head(5).transpose() << std::endl;
    for (int i = 0; i < 10; ++i)
      std::cout <<  theta_est(i, i) << " ";
    std::cout << std::endl;

    for (int i = 0; i < 10; ++i)
      std::cout <<  theta_act(i, i) << " ";
    std::cout << std::endl;
    // std::cout << theta_act.head(5).transpose() << std::endl;
    std::cout << parameter_error.back() << std::endl;
  }

  std::ofstream outfile;
  outfile.open ("data_beam.csv");

  // outfile << "timestamp,theta_est_1,theta_est_2,theta_act_1,theta_act_2,y1,y2,y3,y4" << std::endl;
  outfile << "timestamp";
  std::stringstream s;
  for (int i = 0; i < 182; ++i)
  {
    s << ",theta_est_";
    s << std::setfill('0') << std::setw(3) << i + 1;
  }
  for (int i = 0; i < 182; ++i)
  {
    s << ",theta_act_";
    s << std::setfill('0') << std::setw(3) << i + 1;
  }
  for (int i = 0; i < 13; ++i)
  {
    s << ",q_";
    s << std::setfill('0') << std::setw(3) << i + 1;
  }
  for (int i = 0; i < 13; ++i)
  {
    s << ",dq_";
    s << std::setfill('0') << std::setw(3) << i + 1;
  }
  for (int i = 0; i < 13; ++i)
  {
    s << ",ddq_";
    s << std::setfill('0') << std::setw(3) << i + 1;
  }
  for (int i = 0; i < 13; ++i)
  {
    s << ",F_";
    s << std::setfill('0') << std::setw(3) << i + 1;
  }
  s << ",parameter_error";
  s << ",residual_error";

  // std::cout << "print test " << sprintf("%02d", 2 + 1) << std::endl;
  outfile << s.str() << std::endl;

  for (int i = 0; i < tend; ++i)
  {
    outfile << i * dt;
    // << "," << all_theta_est[i][0] << "," << all_theta_est[i][1]
    //         << "," << theta_true[0] << "," << theta_true[1];
    for (int j = 0; j < 182; ++j)
    {
      outfile << "," << all_theta_est[i][j];
    }

    for (int j = 0; j < 182; ++j)
    {
      outfile << "," << theta_act[j];
    }

    for (int j = 0; j < 13; ++j)
    {
      outfile << "," << all_q[i][j];
    }

    for (int j = 0; j < 13; ++j)
    {
      outfile << "," << all_dq[i][j];
    }

    for (int j = 0; j < 13; ++j)
    {
      outfile << "," << all_ddq[i][j];
    }

    for (int j = 0; j < 13; ++j)
    {
      outfile << "," << all_F[i][j];
    }

    outfile << "," << parameter_error[i];
    outfile << "," << residual_error[i];

    // for (auto & elem : all_y[i])
    //   outfile << "," << elem;

    outfile << std::endl;
  }


  return 1;
}


