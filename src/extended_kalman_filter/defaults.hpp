#ifndef ekf_defaults_hpp
#define ekf_defaults_hpp
#include "extended_kalman_filter.hpp"

namespace mirena::Def
{

    // Symmetric differenciation state predictor
    // Removed the need to manually implement a jacobian
    template <int x_size, int u_size>
    class SDStatePredictor : public StatePredictor<x_size, u_size>
    {
    private:
        // Differentials used for partial derivatives over each variable
        EMatrix<double, x_size, 1> _specific_differencials;

    public:
        constexpr static double DEFAULT_DIFERENTIAL = 0.001;

        EMatrix<double, x_size, x_size> get_jacobian(
            const EMatrix<double, x_size, 1> &state,
            const Eigen::Matrix<double, u_size, 1> &control) override
        {
            EMatrix<double, x_size, x_size> J;
            // Differenciate over each variable of "state" while "control" is constanst
            // We use symmetric derivatves
            for (int j = 0; j < x_size; ++j)
            {
                // Perturb the j-th component by h
                EMatrix<double, x_size, 1> x_plus = state;
                EMatrix<double, x_size, 1> x_minus = state;

                // Check if specific differentials have been set
                double differential;
                if ((differential = this->_specific_differencials(j)) == 0.0)
                {
                    differential = SDStatePredictor::DEFAULT_DIFERENTIAL;
                }

                x_plus(j) += differential;
                x_minus(j) -= differential;

                // Evaluate the function at x + h and x - h
                EMatrix<double, x_size, 1> f_plus = this->predict_state(x_plus, control);
                EMatrix<double, x_size, 1> f_minus = this->predict_state(x_minus, control);

                // Calculate the partial derivatives for all rows i in the Jacobian
                J.col(j) = (f_plus - f_minus) / (2 * differential);
            }
            return J;
        };

        // Set specific pertubation (h) values for each element of the state when applying symmetric differenciation
        // Each element in the passed vector will be the perturbation applied when differenciating respect the correspondent element of the state vector
        // Defaults to 0.001
        void set_specific_differencials(const EMatrix<double, x_size, 1> &differentials)
        {
            this->_specific_differencials = differentials;
        }
    };

    // Symmetric differenciation measure predictor
    // Removed the need to manually implement a jacobian
    template <int x_size, int z_size>
    class SDMeasurePredictor : public MeasurePredictor<x_size, z_size>
    {
    private:
        // Differentials used for partial derivatives over each variable
        EMatrix<double, x_size, 1> _specific_differencials;

    public:
        constexpr static double DEFAULT_DIFERENTIAL = 0.001;

        EMatrix<double, z_size, x_size> get_jacobian(
            const EMatrix<double, x_size, 1> &state) override
        {
            EMatrix<double, z_size, x_size> J;
            // Differenciate over each variable of "state" while "control" is constanst
            // We use symmetric derivatves
            for (int j = 0; j < x_size; ++j)
            {
                // Perturb the j-th component by h
                EMatrix<double, x_size, 1> x_plus = state;
                EMatrix<double, x_size, 1> x_minus = state;

                // Check if specific differentials have been set
                double differential;
                if ((differential = this->_specific_differencials(j)) == 0.0)
                {
                    differential = SDMeasurePredictor::DEFAULT_DIFERENTIAL;
                }

                x_plus(j) += differential;
                x_minus(j) -= differential;

                // Evaluate the function at x + h and x - h
                EMatrix<double, z_size, 1> h_plus = this->predict_measure(x_plus);
                EMatrix<double, z_size, 1> h_minus = this->predict_measure(x_minus);

                // Calculate the partial derivatives for all rows i in the Jacobian
                J.col(j) = (h_plus - h_minus) / (2 * differential);
            }
            return J;
        };

        // Set specific pertubation (h) values for each element of the state when applying symmetric differenciation
        // Each element in the passed vector will be the perturbation applied when differenciating respect the correspondent element of the state vector
        // Defaults to 0.001
        void set_specific_differencials(const EMatrix<double, x_size, 1> &differentials)
        {
            this->_specific_differencials = differentials;
        }
    };
}

#endif