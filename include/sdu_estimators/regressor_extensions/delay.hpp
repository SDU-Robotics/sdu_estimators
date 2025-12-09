#pragma once

#ifndef DELAY_HPP
#define DELAY_HPP

#include "sdu_estimators/regressor_extensions/regressor_extension.hpp"
#include "sdu_estimators/state_estimators/state_space_model.hpp"

#include "sdu_estimators/typedefs.hpp"

#include <deque>

namespace sdu_estimators::regressor_extensions
{
  template <typename T, int32_t DIM_N, int32_t DIM_P>
  class Delay : public RegressorExtension<T, DIM_N, DIM_P>
  {
    static_assert(DIM_N == 1, "The LTI regressor extension only works with N == 1.");

  public:
    /**
     *
     * @param d : Number of samples to delay.
     */
    Delay(std::vector<int> d)
    {
      this->d = d;

      d_max = 0;
      for (int i = 0; i < this->d.size(); ++i)
      {
        if (this->d[i] > d_max)
          d_max = this->d[i];
      }

      // std::cout << d_max << std::endl;

      Eigen::Vector<T, DIM_N> tmp_vec;
      tmp_vec.setZero();

      Eigen::Matrix<T, DIM_P, DIM_N> tmp_mat;
      tmp_mat.setZero();

      // memory_bank_y.empty();
      // memory_bank_phi.empty();

      for (int i = 0; i < d_max + 1; ++i)
      {
        memory_bank_y.push_back(tmp_vec);
        memory_bank_phi.push_back(tmp_mat);
      }

      reset();
    }

    void step(const Eigen::Matrix<T, DIM_N, 1> &y, const Eigen::Matrix<T, DIM_P, DIM_N> &phi)
    {
      Eigen::Vector<T, DIM_N> tmp_vec;
      tmp_vec.setZero();

      Eigen::Matrix<T, DIM_P, DIM_N> tmp_mat;
      tmp_mat.setZero();

      // Push to end of memory bank and pop the first.
      memory_bank_y.pop_back();
      memory_bank_y.push_front(y);

      memory_bank_phi.pop_back();
      memory_bank_phi.push_front(phi);

      // for (int i = 0; i < d_max; ++i)
      // {
      //   std::cout << memory_bank_y.at(i) << ", ";
      // }
      // std::cout << std::endl;

      for (int i = 0; i < DIM_P; ++i)
      {
        // this->y_f(i) = memory_bank_y.at(d[d_max - i - 1]);
        this->y_f(i) = memory_bank_y[d[i]][0];
        // this->phi_f(Eigen::all, i) = memory_bank_phi[d[i] - 1];
        this->phi_f(i, Eigen::all) = memory_bank_phi[d[i]];
      }

      // std::cout << "y_f\n" << this->y_f << std::endl;
      // std::cout << "phi_f\n" << this->phi_f << std::endl;
    }

    void reset()
    {
      this->y_f *= 0;
      this->phi_f *= 0;
    }

  private:
    int d_max;
    std::vector<int> d{};
    std::deque<Eigen::Vector<T, DIM_N>> memory_bank_y;
    std::deque<Eigen::Matrix<T, DIM_P, DIM_N>> memory_bank_phi;
  };
}

#endif //DELAY_HPP
