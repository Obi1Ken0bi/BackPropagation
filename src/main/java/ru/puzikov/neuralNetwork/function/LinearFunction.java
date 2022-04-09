package ru.puzikov.neuralNetwork.function;

public class LinearFunction implements ActivationFunction {
    @Override
    public double activate(double x) {
        return 5 * x + 1;
    }

    @Override
    public double derivate(double x) {
        return 5;
    }
}
