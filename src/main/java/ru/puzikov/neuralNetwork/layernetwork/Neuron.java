package ru.puzikov.neuralNetwork.layernetwork;

import ru.puzikov.neuralNetwork.layernetwork.function.ActivationFunction;

import java.util.Random;

public class Neuron {
    private final ActivationFunction activatorFunction;
    private final double[] weights;
    private double[] inputs;
    private double bias;
    private double output;

    public Neuron(ActivationFunction activatorFunction, double[] weights) {
        this.activatorFunction = activatorFunction;
        this.weights = weights;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double summator() {
        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        sum += bias;
        return sum;
    }

    public void activate() {
        double sum = summator();
        output = activatorFunction.activate(sum);
    }

    public void randomizeWeights(Random rnd) {
        bias = rnd.nextDouble() * 8 - 4;
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rnd.nextDouble() * 8 - 4;
        }
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getOutput() {
        return output;
    }
}
