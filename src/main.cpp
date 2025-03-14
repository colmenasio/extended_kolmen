#include "extended_kalman_filter/extended_kalman_filter.hpp"

#include <iostream>

// Test state model of a 1d particle 
// State is [pos, vel, acc]
// Control is [dt, new_acc]
class TestModel: public StatePredictor<3, 2>{
    public:
    virtual EMatrix<double, 3, 1> predict_state(
        const EMatrix<double, 3, 1> &previous_state,
        const EMatrix<double, 2, 1> &control) override {
            // I KNOW THIS IS A SHIT MODEL LEMME COOOOK
            EMatrix<double, 3, 1> new_state;
            new_state[2] = previous_state[2] + control[1]; // Accel (control 1 sets acceleration)
            new_state[1] = previous_state[1] + previous_state[2] * control[0]; // Speed 
            new_state[0] = previous_state[0] + previous_state[1] * control[0]; // Pos
            return new_state;
        };
};

// Test measure model.
// We expect to observe [2*pos], where pos is the position of a 1d particle
class TestMeasurer: public MeasurePredictor<3, 1>{
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

    auto EKF = ExtendedKalmanFilter<3, 2, 1>(
        std::make_shared<TestModel>(),
        std::make_shared<TestMeasurer>()
    );
    EMatrix<double, 2, 1> control_1_test;
    control_1_test[0] = 1; // 1 second of dt
    control_1_test[1] = 1; // 1 unit of acceleration
    std::cout << EKF.predict_state(control_1_test) << std::endl << std::endl;
    std::cout << EKF.predict_measures(control_1_test) << std::endl << std::endl;
    
    EKF.update(control_1_test, Eigen::Matrix<double, 1, 1>::Ones());

    std::cout << EKF.predict_state(control_1_test) << std::endl << std::endl;
    std::cout << EKF.predict_measures(control_1_test) << std::endl;
    return 0;
}
