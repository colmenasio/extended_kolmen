#ifndef extended_kalman_filter_hpp
#define extended_kalman_filter_hpp

#include <memory>

#include <eigen3/Eigen/Eigen>

#define EMatrix Eigen::Matrix

template <int x_size, int u_size>
class StatePredictor
{
public:
    virtual EMatrix<double, x_size, 1> predict_state(
        const EMatrix<double, x_size, 1> &previous_state,
        const EMatrix<double, u_size, 1> &control) = 0;
};

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
    EMatrix<double, x_size, 1> _state;
    EMatrix<double, x_size, x_size> _state_covariances;

    std::shared_ptr<StatePredictor<x_size, u_size>> _state_predictor;
    std::shared_ptr<MeasurePredictor<x_size, z_size>> _measure_predictor;

    // Config values
    double default_differential = 0.001;
    EMatrix<double, x_size, x_size> _process_covariances;
    EMatrix<double, z_size, z_size> _measure_covariances;

    // Calculate the jacobian of f using symmetric difference 
    EMatrix<double, x_size, x_size> get_F(const Eigen::Matrix<double, u_size, 1> &control)
    {
        EMatrix<double, x_size, x_size> J;
        for (int j = 0; j < x_size; ++j) {
            // Perturb the j-th component by h
            EMatrix<double, x_size, 1> x_plus = this->_state;
            EMatrix<double, x_size, 1> x_minus = this->_state;
            
            x_plus(j) += this->default_differential;
            x_minus(j) -= this->default_differential;
            
            // Evaluate the function at x + h and x - h
            EMatrix<double, x_size, 1> f_plus = this->_state_predictor->predict_state(x_plus, control);
            EMatrix<double, x_size, 1> f_minus = this->_state_predictor->predict_state(x_minus, control);
            
            // Calculate the partial derivatives for all rows i in the Jacobian
            J.col(j) = (f_plus - f_minus) / (2 * this->default_differential);
        }
    
        return J;
    
    };

    // Calculate the jacobian of g using symmetric difference
    EMatrix<double, z_size, x_size> get_H() {
        EMatrix<double, z_size, x_size> J;
        for (int j = 0; j < x_size; ++j) {
            // Perturb the j-th component by h
            EMatrix<double, x_size, 1> x_plus = this->_state;
            EMatrix<double, x_size, 1> x_minus = this->_state;
            
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
    ExtendedKalmanFilter() = delete;
    ExtendedKalmanFilter(
        std::shared_ptr<StatePredictor<x_size, u_size>> state_predictor,
        std::shared_ptr<MeasurePredictor<x_size, z_size>> measure_predictor) : _state_predictor(state_predictor),
                                                                               _measure_predictor(measure_predictor),
                                                                               _state_covariances(EMatrix<double, x_size, x_size>::Identity()),
                                                                               _state(EMatrix<double, x_size, 1>::Zero()),
                                                                               _process_covariances(EMatrix<double, x_size, x_size>::Identity()),
                                                                               _measure_covariances(EMatrix<double, z_size, z_size>::Identity()) {}
    ~ExtendedKalmanFilter() = default;

    EMatrix<double, x_size, 1> predict_state(const Eigen::Matrix<double, u_size, 1> control)
    {
        return this->_state_predictor->predict_state(this->_state, control);
    }
    EMatrix<double, z_size, 1> predict_measures()
    {
        return this->_measure_predictor->predict_measure(this->_state);
    };

    void update(const EMatrix<double, u_size, 1> &control, const Eigen::Matrix<double, z_size, 1> &measure)
    {
        // Bindings
        EMatrix<double, x_size, x_size> F = this->get_F(control);
        EMatrix<double, z_size, x_size> H = this->get_H();

        // Prediction
        EMatrix<double, x_size, 1> predicted_x = this->predict_state(control);
        EMatrix<double, x_size, x_size> predicted_covariance = (F) * (this->_state_covariances) * (F.transpose()) + this->_process_covariances;

        // Update
        EMatrix<double, z_size, 1> measure_residual = measure - this->predict_measures();
        EMatrix<double, z_size, z_size> covariance_residual = (H) * (predicted_covariance) * (H.transpose()) + this->_measure_covariances;
        EMatrix<double, x_size, z_size> K_gain = (predicted_covariance) * (H.transpose()) * (covariance_residual.inverse());
        this->_state = predicted_x + (K_gain) * (measure_residual);
        this->_state_covariances = (EMatrix<double, x_size, x_size>::Identity() - K_gain * H)*(predicted_covariance);
    };
};

#endif