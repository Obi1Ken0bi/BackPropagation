package ru.puzikov.neuralNetwork;

import ru.puzikov.neuralNetwork.layernetwork.LayerNetwork;
import ru.puzikov.neuralNetwork.layernetwork.function.ActivationFunction;
import ru.puzikov.neuralNetwork.layernetwork.function.LinearFunction;
import ru.puzikov.neuralNetwork.layernetwork.function.SigmoidFunction;
import ru.puzikov.neuralNetwork.layernetwork.function.TanhFunction;

import java.util.Arrays;

public class NeuralNetwork {
    public static void main(String[] args) {
        pictureProcessing();

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
        ActivationFunction linearFunction = new LinearFunction();
        ActivationFunction tanh = new TanhFunction();
        LayerNetwork network = new LayerNetwork(new int[]{15, 3},
                inputs[0].length,
                new ActivationFunction[]{tanh, activationFunction});
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
