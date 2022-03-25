package ru.puzikov.neuralNetwork;

import lombok.Getter;
import lombok.Setter;
import ru.puzikov.neuralNetwork.function.ActivationFunction;

import java.util.Random;

@Getter
@Setter
public class LayerNetwork {
    private Layer[] hiddenLayers;
    private double[] inputLayer;
    private double[] output;
    private double mse;

    public LayerNetwork(int[] sizes, int[] previousLayerSizes, ActivationFunction[] activationFunctions) {
        hiddenLayers = new Layer[sizes.length];
        fillLayersList(sizes, previousLayerSizes, activationFunctions);

    }

    public void compute() {
        hiddenLayers[0].setInputs(inputLayer);
        double[] output = hiddenLayers[0].compute();
        for (int i = 1; i < hiddenLayers.length; i++) {
            hiddenLayers[i].setInputs(output);
            output = hiddenLayers[i].compute();
        }
        this.output = output;
    }

    public void randomizeWeights() {
        Random rnd = new Random();
        for (Layer hiddenLayer : hiddenLayers) {
            hiddenLayer.randomizeWeights(rnd);
        }
    }

    private void fillLayersList(int[] sizes, int[] previousLayerSizes, ActivationFunction[] activationFunctions) {
        for (int i = 0; i < sizes.length; i++) {
            hiddenLayers[i] = new Layer(sizes[i], previousLayerSizes[i], activationFunctions[i]);
        }
    }

    public void train(int epochCount, double learningRate, double[][] inputs, double[][] outputs) {
        double[] mses = new double[inputs.length];
        randomizeWeights();
        for (int i = 0; i < epochCount; i++) {
            double percentage = (double) i / epochCount * 100;
            System.out.println("Progress: " + percentage + "% " + "Error = " + mse);

            for (int j = 0; j < inputs.length; j++) {
                double[] trainInput = inputs[j];
                double[] trainOutput = outputs[j];
                inputLayer = trainInput;
                compute();
                mses[j] = getOneTrainMse(trainOutput, output);
                backPropagation(trainOutput, learningRate);
            }
            mse = getEpochMSE(mses);
        }
        System.out.println();
    }

    public void backPropagation(double[] trainOutput, double learningRate) {
        double[] error = new double[trainOutput.length];
        for (int i = 0; i < error.length; i++) {
            error[i] = output[i] - trainOutput[i];
        }
        for (int i = hiddenLayers.length - 1; i >= 0; i--) {
            double[] previousLayerOutputs = i == 0 ? inputLayer : hiddenLayers[i - 1].getOutputs();
            error = hiddenLayers[i].computeError(error, previousLayerOutputs, learningRate);
        }

    }

    private double getOneTrainMse(double[] trainOutput, double[] actualOutput) {
        double mse = 0;
        for (int i = 0; i < trainOutput.length; i++) {
            mse += Math.pow(actualOutput[i] - trainOutput[i], 2);
        }
        return mse;
    }

    private double getEpochMSE(double[] mses) {
        double mse = 0;
        for (double ms : mses) {
            mse += ms;
        }
        return mse / mses.length;
    }


}
