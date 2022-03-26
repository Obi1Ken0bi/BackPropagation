package ru.puzikov.neuralNetwork.function;

public class LinearFunction implements ActivationFunction{
    @Override
    public double activate(double x) {
        return 12*x+2;
    }

    @Override
    public double derivate(double x) {
        return 12;
    }
}
