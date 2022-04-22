package ru.puzikov.neuralNetwork.layer;

import ru.puzikov.neuralNetwork.neuron.NeuronRadial;

public class LayerRadial {
    private final NeuronRadial[] neurons;
    private double[] output;

    public LayerRadial(int neuronCount, double[][] coefficients, double s){
        neurons=new NeuronRadial[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            neurons[i]=new NeuronRadial(s,coefficients[i]);
        }
    }

    public double[] compute(double[] inputs){
        double[] res=new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            res[i]=neurons[i].activate(inputs);
        }
        output=res;
        return output;

    }

    public double[] getOutput() {
        return output;
    }

    public void setOutput(double[] output) {
        this.output = output;
    }
}
