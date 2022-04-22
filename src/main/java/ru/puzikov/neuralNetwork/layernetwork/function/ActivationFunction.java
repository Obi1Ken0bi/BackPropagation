package ru.puzikov.neuralNetwork.layernetwork.function;

public interface ActivationFunction {
    double activate(double x);

    double derivate(double x);
}
