package ru.puzikov.neuralNetwork;

import ru.puzikov.neuralNetwork.function.ActivationFunction;
import ru.puzikov.neuralNetwork.function.SigmoidFunction;
import ru.puzikov.neuralNetwork.network.LayerNetwork;
import ru.puzikov.neuralNetwork.network.RadialNetwork;

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
        ActivationFunction activationFunction = new SigmoidFunction();
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
    private static void pictureProcessing() {
        double[][] outputs = new double[][]{
                new double[]{0, 0, 1},
                new double[]{0, 0, 1},
                new double[]{1, 0, 0},
                new double[]{1, 0, 0},
                new double[]{1, 0, 0},
                new double[]{1, 0, 0},
                new double[]{1, 0, 0},
                new double[]{0, 1, 0},
                new double[]{0, 1, 0},
                new double[]{0, 1, 0},
                new double[]{0, 1, 0},
                new double[]{0, 1, 0}
        };


        double[][] inputs = new double[][]{
                new double[]{
                        0, 0, 1, 1, 1,
                        0, 0, 0, 1, 0,
                        0, 0, 1, 0, 0,
                        0, 1, 0, 0, 0,
                        1, 1, 1, 1, 0
                },
                new double[]{
                        0, 1, 1, 1, 1,
                        0, 0, 0, 1, 0,
                        0, 0, 1, 0, 0,
                        0, 1, 0, 0, 0,
                        1, 1, 1, 1, 1
                },
                new double[]{
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        1, 1, 1, 1, 1
                },
                new double[]{
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        1, 1, 1, 1, 1,
                        0, 0, 0, 0, 0
                },
                new double[]{
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 1, 1, 1
                },
                new double[]{
                        1, 1, 1, 1, 1,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0
                },
                new double[]{
                        1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0
                },
                new double[]{
                        0, 1, 0, 0, 0,
                        1, 1, 1, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0
                },
                new double[]{
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        1, 1, 1, 0, 0,
                        0, 1, 0, 0, 0
                },
                new double[]{
                        0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        1, 1, 1, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 0, 0, 0, 0
                },
                new double[]{
                        0, 0, 0, 0, 0,
                        0, 0, 1, 0, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0
                },
                new double[]{
                        0, 0, 1, 0, 0,
                        0, 0, 1, 0, 0,
                        1, 1, 1, 1, 1,
                        0, 0, 1, 0, 0,
                        0, 0, 1, 0, 0
                }
        };


        ActivationFunction activationFunction = new SigmoidFunction();

        LayerNetwork network = new LayerNetwork(new int[]{15, 3},
                inputs[0].length,
                new ActivationFunction[]{new SigmoidFunction(), activationFunction});
        network.train(50000, 0.1, inputs, outputs);
        for (int i = 0; i < inputs.length; i++) {
            network.setInputLayer(inputs[i]);
            network.compute();
            System.out.println("Output: " + Arrays.toString(network.getOutput()) + " Expected: " + Arrays.toString(outputs[i]));
        }
        network.setInputLayer(new double[]{
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 1, 0,
                0, 0, 1, 1, 1,
                0, 0, 0, 1, 0
        });
        network.compute();
        System.out.println(Arrays.toString(network.getOutput()));
        network.setInputLayer(new double[]{
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                1, 1, 1, 1, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0
        });
        network.compute();
        System.out.println(Arrays.toString(network.getOutput()));
        network.setInputLayer(new double[]{
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 0
        });
        network.compute();
        System.out.println(Arrays.toString(network.getOutput()));
        network.setInputLayer(new double[]{
                0, 1, 1, 1, 1,
                0, 0, 0, 1, 0,
                0, 0, 1, 0, 0,
                0, 0.7, 0, 0, 0,
                1, 1, 1, 1, 1
        });
        network.compute();
        System.out.println(Arrays.toString(network.getOutput()));
    }
}
