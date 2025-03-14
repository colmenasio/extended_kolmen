#ifndef extended_kalman_filter_hpp
#define extended_kalman_filter_hpp

#include <memory>

#include <eigen3/Eigen/Eigen>

#define EMatrix Eigen::Matrix

#define DEFAULT_DIFERENTIAL 0.001

// State Prediction Function Abstract class
//
// Concrete models should inherit from this class
// Corresponds to the "f" function of the kalman filter, predicting the next state in function of the previous state and current control
template <int x_size, int u_size>
class StatePredictor
{
public:
    virtual EMatrix<double, x_size, 1> predict_state(
        const EMatrix<double, x_size, 1> &previous_state,
        const EMatrix<double, u_size, 1> &control) = 0;
};

// Measure Prediction Function Abstract class
//
// Concrete models should inherit from this class
// Corresponds to the "h" function of the kalman filter, predicting the measure in function of the current state
template <int x_size, int z_size>
class MeasurePredictor
{
public:
    virtual EMatrix<double, z_size, 1> predict_measure(
        const EMatrix<double, x_size, 1> &state) = 0;
};

// Generic Implementation of an Extended Kalman Filter
//
// [ TEMPLATE GENERICS ]
// "int x_size" is the size in doubles of the state vector
// "int u_size" is the size in doubles of the control vector
// "int z_size" is the size in doubles of the measure vector
//
// [ USAGE ]
// A EKF requires defining a State vector, Control vector, Measure vector
// To do this, provide the size in doubles of each one in the template (x_size, u_size, z_size)
//
// Additionaly, a State Prediction and a Measure Prediction functions of matching sizes are required.
// They must inherit from StatePredictor<x_size, u_size> and MeasurePredictor<x_size, z_size> respectively
template <int x_size, int u_size, int z_size>
class ExtendedKalmanFilter
{
private:
    // State and others
    EMatrix<double, x_size, 1> _state;
    EMatrix<double, x_size, x_size> _state_covariances;

    // Predictors (state and measure models)
    std::shared_ptr<StatePredictor<x_size, u_size>> _state_predictor;
    std::shared_ptr<MeasurePredictor<x_size, z_size>> _measure_predictor;
 
    // Config values
    EMatrix<double, x_size, 1> _specific_differencials;
    EMatrix<double, x_size, x_size> _process_covariances;
    EMatrix<double, z_size, z_size> _measure_covariances;

    // Calculate the jacobian of f using symmetric difference
    EMatrix<double, x_size, x_size> get_F(const Eigen::Matrix<double, u_size, 1> &control)
    {
        EMatrix<double, x_size, x_size> J;
        // Differenciate over each variable of "state" while "control" is constanst
        // We use symmetric derivatves
        for (int j = 0; j < x_size; ++j)
        {
            // Perturb the j-th component by h
            EMatrix<double, x_size, 1> x_plus = this->_state;
            EMatrix<double, x_size, 1> x_minus = this->_state;

            x_plus(j) += this->_specific_differencials(j);
            x_minus(j) -= this->_specific_differencials(j);

            // Evaluate the function at x + h and x - h
            EMatrix<double, x_size, 1> f_plus = this->_state_predictor->predict_state(x_plus, control);
            EMatrix<double, x_size, 1> f_minus = this->_state_predictor->predict_state(x_minus, control);

            // Calculate the partial derivatives for all rows i in the Jacobian
            J.col(j) = (f_plus - f_minus) / (2 * this->default_differential);
        }
        return J;
    };

    // Calculate the jacobian of g using symmetric difference
    EMatrix<double, z_size, x_size> get_H(const Eigen::Matrix<double, x_size, 1> &predicted_state)
    {
        EMatrix<double, z_size, x_size> J;
        for (int j = 0; j < x_size; ++j)
        {
            // Perturb the j-th component by h
            EMatrix<double, x_size, 1> x_plus = predicted_state;
            EMatrix<double, x_size, 1> x_minus = predicted_state;

            x_plus(j) += this->default_differential;
            x_minus(j) -= this->default_differential;

            // Evaluate the function at x + h and x - h
            EMatrix<double, z_size, 1> f_plus = this->_measure_predictor->predict_measure(x_plus);
            EMatrix<double, z_size, 1> f_minus = this->_measure_predictor->predict_measure(x_minus);

            // Calculate the partial derivatives for all rows i in the Jacobian
            J.col(j) = (f_plus - f_minus) / (2 * this->default_differential);
        }

        return J;
    };

public:
    // Default constructor disabled. Provide a state_predictor and a measure_predictor of appropiate size instead.
    ExtendedKalmanFilter() = delete;

    // Intended constuctor
    // Provide instance of clases implementing the StatePredictor and MeasurePredictor appropiately
    ExtendedKalmanFilter(
        std::shared_ptr<StatePredictor<x_size, u_size>> state_predictor,
        std::shared_ptr<MeasurePredictor<x_size, z_size>> measure_predictor) : _state_predictor(state_predictor),
                                                                               _measure_predictor(measure_predictor),
                                                                               _state_covariances(EMatrix<double, x_size, x_size>::Identity()),
                                                                               _state(EMatrix<double, x_size, 1>::Zero()),
                                                                               _process_covariances(EMatrix<double, x_size, x_size>::Identity()),
                                                                               _measure_covariances(EMatrix<double, z_size, z_size>::Identity())
    {
        this->_specific_differencials.setConstant(DEFAULT_DIFERENTIAL)
    }

    ~ExtendedKalmanFilter() = default;

    // Update the state of the kalman filter with new measures
    void update(const EMatrix<double, u_size, 1> &control, const Eigen::Matrix<double, z_size, 1> &measure)
    {
        // Bindings (predictions and jacobians)
        EMatrix<double, x_size, 1> predicted_state = this->predict_state(control);
        EMatrix<double, z_size, 1> predicted_measure = this->_measure_predictor->predict_measure(predicted_state);
        EMatrix<double, x_size, x_size> F = this->get_F(control);
        EMatrix<double, z_size, x_size> H = this->get_H(predicted_state);

        // Prediction
        EMatrix<double, x_size, x_size> predicted_covariance = (F) * (this->_state_covariances) * (F.transpose()) + this->_process_covariances;

        // Update
        EMatrix<double, z_size, 1> measure_residual = measure - predicted_measure; // TODO fix this bug, this predicts over old state not predicted state
        EMatrix<double, z_size, z_size> covariance_residual = (H) * (predicted_covariance) * (H.transpose()) + this->_measure_covariances;
        EMatrix<double, x_size, z_size> K_gain = (predicted_covariance) * (H.transpose()) * (covariance_residual.inverse());
        this->_state = predicted_state + (K_gain) * (measure_residual);
        this->_state_covariances = (EMatrix<double, x_size, x_size>::Identity() - K_gain * H) * (predicted_covariance);
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // Public interface. 
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    // Predict the state according to the old state and some control input
    EMatrix<double, x_size, 1> predict_state(const Eigen::Matrix<double, u_size, 1> control)
    {
        return this->_state_predictor->predict_state(this->_state, control);
    }

    // Predict the measures according to the old state and some control input
    EMatrix<double, z_size, 1> predict_measures(const Eigen::Matrix<double, u_size, 1> control)
    {
        return this->_measure_predictor->predict_measure(this->predict_state(control));
    };

    // Set specific pertubation (h) values for each element of the state when applying symmetric differenciation
    // Each element in the passed vector will be the perturbation applied when differenciating respect the correspondent element of the state vector
    // Defaults to 0.001 
    void set_specific_differencials(const EMatrix<double, x_size, 1>& differentials){
        this->_specific_differencials = differentials;
    }

    // Set the covariance matrix of the state prediction noise (Q)
    void set_process_covariance(EMatrix<double, x_size, x_size>& process_covariances){
        this->_process_covariances = process_covariances;
    }

    // Set the covariance matrix of the measure prediction noise (R)
    void set_measure_covariance(EMatrix<double, z_size, z_size>& measure_covariances){
        this->_measure_covariances = measure_covariances;
    }

};

#endif