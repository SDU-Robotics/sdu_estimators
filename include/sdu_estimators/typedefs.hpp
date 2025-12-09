#pragma once
#ifndef TYPEDEFS_HPP
#define TYPEDEFS_HPP

#include <Eigen/Core>

#if (EIGEN_WORLD_VERSION == 3)
    #if (EIGEN_MAJOR_VERSION < 4)
        namespace Eigen 
        {
            template <typename Type, int Size>
            using Vector = Matrix<Type, Size, 1>;
            
            using all = indexing::all;
        }
    #endif 
#endif

#endif   // TYPEDEFS_HPP