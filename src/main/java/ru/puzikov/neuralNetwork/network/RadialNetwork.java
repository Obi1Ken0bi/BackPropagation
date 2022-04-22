package ru.puzikov.neuralNetwork.network;

import ru.puzikov.neuralNetwork.function.ActivationFunction;
import ru.puzikov.neuralNetwork.layer.Layer;
import ru.puzikov.neuralNetwork.layer.LayerRadial;

import java.util.Arrays;

public class RadialNetwork {
    private final LayerRadial layerRadial;
    private final Layer outputLayer;
    private final double[][] targets;
    private final double[][] inputs;

    public RadialNetwork(double[][] inputs, double[][] results,double s){
        this.inputs=inputs;
        targets=results;
        layerRadial=new LayerRadial(inputs.length,inputs,s);
        outputLayer=new Layer(targets[0].length, inputs.length, new ActivationFunction() {
            @Override
            public double activate(double x) {
                return x;
            }

            @Override
            public double derivate(double x) {
                return 1;
            }
        });

    }

    public double[] getResult(double[] input){
        double[] output=layerRadial.compute(input);
        outputLayer.setInputs(output);
        return outputLayer.compute();
    }

    public void train(int epochCount, double learningRate){
    
        for (int i = 0; i < epochCount; i++) {
             double error=trainNetwork(learningRate);
            System.out.printf("Progress: %d/%d.\tError: %s%n", i, epochCount, error);
        }
        for (int i = 0; i < inputs.length; i++) {
            double[] netResult=this.getResult(inputs[i]);
            String input = Arrays.toString(inputs[i]);
            String result = Arrays.toString(netResult);
            String expected = Arrays.toString(targets[i]);
            System.out.printf("For input: %s    got: %s    expected: %s",input,result,expected);

        }
    }

    private double trainNetwork(double learningRate) {
        double error=0d;
        for (int i = 0; i < targets.length; i++) {
            this.getResult(inputs[i]);
            for (int j = 0; j < outputLayer.getNeurons().length; j++) {
                error+=trainNeuron(learningRate,i,j);
            }
        }
        return error;
    }

    private double trainNeuron(double learningRate, int i, int j) {
        var xK = layerRadial.getOutput();
        var yk = targets[i][j];

        var neuron = outputLayer.getNeurons()[j];
        neuron.setInputs(xK);
        neuron.activate();
        var oK=neuron.getOutput();

        var ek=0.5d*Math.pow(yk-oK,2);

        for (int z = 0; i < neuron.getWeights().length; z++)
        {
            neuron.getWeights()[z] = neuron.getWeights()[z] - learningRate * (yk - oK) * (-xK[z]);
        }

        return ek;
    }
}
