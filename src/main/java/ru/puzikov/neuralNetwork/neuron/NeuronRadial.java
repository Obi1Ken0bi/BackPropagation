package ru.puzikov.neuralNetwork.neuron;

public class NeuronRadial   {
    private final double s;
    private double[] coefficients;
    private double value;

    public NeuronRadial(double s, double[] coefficients) {
        this.s = s;
        this.coefficients = coefficients;
    }

    public double activate(double[] inputs){
        value=Math.exp(-(Normalize(inputs)*(0.8236/s)));
        return value;
    }

    private double Normalize(double[] inputs) {
        double res=0;
        for (int i = 0; i < coefficients.length; i++) {
            res+=Math.pow(inputs[i]-coefficients[i], 2);
        }
        return Math.sqrt(res);
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }
}
