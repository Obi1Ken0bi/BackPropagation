package ru.puzikov.neuralNetwork.layernetwork.function;

public class SigmoidFunction implements ActivationFunction {
    @Override
    public double activate(double x) {
        return 1 / (1 + Math.pow(Math.E, -x));
    }

    @Override
    public double derivate(double x) {
        double f = activate(x);
        return f * (1 - f);
    }
}
