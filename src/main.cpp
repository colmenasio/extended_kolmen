#include "extended_kalman_filter/extended_kalman_filter.hpp"
#include "extended_kalman_filter/defaults.hpp"

#include <iostream>

// EXAMPLE OF THE USE OF THE KALMAN FILTER USING A 1-d PARTICLE MODEL

// Test state model of a 1d particle 
// State is [pos, vel, acc]
// Control is [dt, new_acc]
class TestModel: public Def::SDStatePredictor<3, 2>{
    public:
    virtual EMatrix<double, 3, 1> predict_state(
        const EMatrix<double, 3, 1> &previous_state,
        const EMatrix<double, 2, 1> &control) override {
            // I KNOW THIS IS A SHIT MODEL LEMME COOOOK
            EMatrix<double, 3, 1> new_state ;

            // Acceleration
            new_state[2] = control[1];

            // Speed
            new_state[1] = previous_state[1] + (previous_state[2] + new_state[2])/2 * control[0];

            // Pos
            new_state[0] = previous_state[0] + (previous_state[1] + new_state[1])/2 * control[0];

            return new_state;
        };
};

// Test measure model.
// We expect to observe [2*pos], where pos is the position of a 1d particle
class TestMeasurer: public Def::SDMeasurePredictor<3, 1>{
    public:
    virtual EMatrix<double, 1, 1> predict_measure(
        const EMatrix<double, 3, 1> &state) override{
            EMatrix<double, 1, 1> measure;
            measure[0] = 2*state[0]; // We observe a value proportional to the position
            return measure;
        };
};

int main(int argc, char const *argv[])
{
    // Instanciate the filter
    auto EKF = ExtendedKalmanFilter<3, 2, 1>(
        std::make_shared<TestModel>(),
        std::make_shared<TestMeasurer>()
    );

    // Make a dummy control vector vector
    EMatrix<double, 2, 1> control_1_test;
    control_1_test[0] = 1; // 1 second of dt
    control_1_test[1] = 1; // 1 unit of acceleration

    // Predictions before and after the state update
    std::cout << EKF.predict_state(control_1_test) << std::endl << std::endl;
    std::cout << EKF.predict_measures(control_1_test) << std::endl << std::endl;
    
    EKF.update(control_1_test, Eigen::Matrix<double, 1, 1>::Ones());

    std::cout << EKF.predict_state(control_1_test) << std::endl << std::endl;
    std::cout << EKF.predict_measures(control_1_test) << std::endl;
    return 0;
}
