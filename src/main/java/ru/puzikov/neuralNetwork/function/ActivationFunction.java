package ru.puzikov.neuralNetwork.function;

public interface ActivationFunction {
    double activate(double x);

    double derivate(double x);
}
