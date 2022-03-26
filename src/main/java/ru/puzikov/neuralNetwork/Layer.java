package ru.puzikov.neuralNetwork;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;
import lombok.Setter;
import ru.puzikov.neuralNetwork.function.ActivationFunction;

import java.util.Arrays;
import java.util.Random;

@Getter
@Setter
public class Layer {
    private Neuron[] neurons;
    private int size;
    private int previousLayersSize;
    private double[] outputs;
    @JsonIgnore
    private ActivationFunction activationFunction;

    public Layer(int size, int previousLayersSize, ActivationFunction activationFunction) {
        this.size = size;
        this.previousLayersSize = previousLayersSize;
        this.activationFunction = activationFunction;
        fillNeuronsList();
    }

    public double[] compute() {
        double[] output = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].activate();
            output[i] = neurons[i].getOutput();
        }
        return output;
    }

    public void randomizeWeights(Random rnd) {
        for (Neuron neuron : neurons) {
            neuron.randomizeWeights(rnd);
        }
    }

    private void fillNeuronsList() {
        neurons = new Neuron[size];
        for (int i = 0; i < neurons.length; i++) {
            double[] weights = new double[previousLayersSize];
            neurons[i] = new Neuron(activationFunction, weights);
        }
    }

    public double[] computeError(double[] errors, double[] previousLayerOutputs, double learningRate) {
        double[] weightsDeltas = new double[errors.length];
        double[] outputErrors = new double[previousLayersSize];
        for (int i = 0; i < neurons.length; i++) {
            weightsDeltas[i] = getWeightsDelta(errors[i]);
            for (int j = 0; j < neurons[i].getWeights().length; j++) {
                neurons[i].getWeights()[j] = neurons[i].getWeights()[j] - previousLayerOutputs[j] * weightsDeltas[i] * learningRate;
                outputErrors[j] += weightsDeltas[i] * neurons[i].getWeights()[j];
            }
            neurons[i].setBias(neurons[i].getBias() - weightsDeltas[i] * learningRate);
        }
        return outputErrors;
    }

    private double getWeightsDelta(double error) {
        return error * activationFunction.derivate(error);
    }

    public double[] getOutputs() {
        return Arrays.stream(neurons).mapToDouble(Neuron::getOutput).toArray();
    }

    public void setInputs(double[] inputs) {
        for (Neuron neuron : neurons) {
            neuron.setInputs(inputs);
        }
    }
}
