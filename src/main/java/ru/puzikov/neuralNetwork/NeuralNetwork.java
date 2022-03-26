package ru.puzikov.neuralNetwork;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import ru.puzikov.neuralNetwork.function.ActivationFunction;
import ru.puzikov.neuralNetwork.function.LinearFunction;
import ru.puzikov.neuralNetwork.function.SigmoidFunction;

import java.io.File;
import java.io.IOException;

public class NeuralNetwork {
    public static void main(String[] args) {
        double[][] inputs = new double[1000][];


        double[][] outputs = new double[1000][];

        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = new double[]{i};
            outputs[i] = new double[]{Math.sqrt(i)};
        }
        ActivationFunction activationFunction = new SigmoidFunction();
        ActivationFunction linearFunction = new LinearFunction();
        LayerNetwork network = new LayerNetwork(new int[]{15, 1},
                inputs[0].length,
                new ActivationFunction[]{linearFunction, linearFunction});
        network.train(100000, 0.000000000001, inputs, outputs);
        for (int i = 0; i < inputs.length; i++) {
            network.setInputLayer(inputs[i]);
            network.compute();
            System.out.println("Output: " + network.getOutput()[0] + " Expected: " + outputs[i][0]);
        }
        network.setInputLayer(new double[]{16});
        network.compute();
        System.out.println(network.getOutput()[0]);
        network.setInputLayer(new double[]{49});
        network.compute();
        System.out.println(network.getOutput()[0]);
        ObjectMapper jsonMapper = new JsonMapper();
        try {
            jsonMapper.writeValue(new File("Test.json"), network);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
