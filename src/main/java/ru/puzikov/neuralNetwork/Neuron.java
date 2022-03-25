package ru.puzikov.neuralNetwork;

import ru.puzikov.neuralNetwork.function.ActivationFunction;

import java.util.Random;

public class Neuron {
    private final ActivationFunction activatorFunction;
    private double[] inputs;
    private final double[] weights;
    private double bias;
    private double output;

    public Neuron(ActivationFunction activatorFunction, double[] weights) {
        this.activatorFunction = activatorFunction;
        this.weights = weights;
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
