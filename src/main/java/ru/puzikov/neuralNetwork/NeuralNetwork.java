package ru.puzikov.neuralNetwork;

import ru.puzikov.neuralNetwork.function.ActivationFunction;
import ru.puzikov.neuralNetwork.function.Sigmoid;
import ru.puzikov.neuralNetwork.network.LayerNetwork;

import java.util.Arrays;

public class NeuralNetwork {
    public static void main(String[] args) {
        double[][] inputs =
                {
                        new double [] { 0.0,0.0 },
                        new double [] { 0.0,1.0 },
                        new double [] { 1.0, 0.0d},
                        new double [] { 1.0d, 1.0},
                };

        double[][] outputs =
                {
                        new double[] { 0.0d },
                        new double[] { 0.0d },
                        new double[] { 0.0d },
                        new double[] { 1.0d },
                };
        ActivationFunction activationFunction = new Sigmoid();
        LayerNetwork network = new LayerNetwork(new int[]{25,25, 1},
                2,
                new ActivationFunction[]{activationFunction, activationFunction,activationFunction});
        network.train(10000, 0.01, inputs, outputs);
        for (int i = 0; i < inputs.length; i++) {
            network.setInputLayer(inputs[i]);
            network.compute();
            System.out.println("Output: " + network.getOutput()[0] + " Expected: " + outputs[i][0]);
        }
    }
}
