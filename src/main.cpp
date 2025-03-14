#include "extended_kalman_filter/extended_kalman_filter.hpp"

class TestModel: public StatePredictor<3, 1>{
    public:
    virtual EMatrix<double, 3, 1> predict_state(
        const EMatrix<double, 3, 1> &previous_state,
        const EMatrix<double, 1, 1> &control) override {
            // I KNOW THIS IS A SHIT MODEL LEMME COOOOK
            EMatrix<double, 3, 1> new_state;
            new_state[2] = previous_state[2]; // Accel
            new_state[1] = previous_state[1] + previous_state[2] * control[0]; // Speed 
            new_state[0] = previous_state[0] + previous_state[1] * control[0]; // Pos
            return new_state;
        };
};

class TestMeasurer: public MeasurePredictor<3, 1>{
    public:
    virtual EMatrix<double, 1, 1> predict_measure(
        const EMatrix<double, 3, 1> &state) override{
            return EMatrix<double, 1, 1>(state[0]);
        };
};

int main(int argc, char const *argv[])
{

    auto EKF = ExtendedKalmanFilter<3, 1, 1>(
        std::make_shared<TestModel>(),
        std::make_shared<TestMeasurer>()
    ); 
    return 0;
}
