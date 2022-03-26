package ru.puzikov.neuralNetwork;

import ru.puzikov.neuralNetwork.function.ActivationFunction;
import ru.puzikov.neuralNetwork.function.LinearFunction;
import ru.puzikov.neuralNetwork.function.SigmoidFunction;

public class NeuralNetwork {
    public static void main(String[] args) {
        double[][] inputs =new double[100][];
//                {
//                        new double [] { 0.0,0.0 },
//                        new double [] { 0.0,1.0 },
//                        new double [] { 1.0, 0.0d},
//                        new double [] { 1.0d, 1.0},
//                };

        double[][] outputs =new double[100][];
//                {
//                        new double[] { 0.0d },
//                        new double[] { 0.0d },
//                        new double[] { 0.0d },
//                        new double[] { 1.0d },
//                };
        for (int i = 0; i < 100; i++) {
            inputs[i]=new double[]{i*10};
            outputs[i]=new double[]{2*i*10};
        }
        ActivationFunction activationFunction = new SigmoidFunction();
        ActivationFunction linearFunction=new LinearFunction();
        LayerNetwork network = new LayerNetwork(new int[]{5,1},
                inputs[0].length,
                new ActivationFunction[]{linearFunction,linearFunction});
        network.train(500000, 0.000000000001, inputs, outputs);
        for (int i = 0; i < inputs.length; i++) {
            network.setInputLayer(inputs[i]);
            network.compute();
            System.out.println("Output: " + network.getOutput()[0] + " Expected: " + outputs[i][0]);
        }
        network.setInputLayer(new double[]{3000});
        network.compute();
        System.out.println(network.getOutput()[0]);
    }
}
