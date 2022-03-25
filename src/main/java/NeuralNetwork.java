import ru.puzikov.neuralNetwork.LayerNetwork;
import ru.puzikov.neuralNetwork.function.ActivationFunction;
import ru.puzikov.neuralNetwork.function.Sigmoid;

public class NeuralNetwork {
    public static void main(String[] args) {
        double[][] inputs =
                {
                        new double[]{0.0d, 0.0d, 0.0d},
                        new double[]{0.0d, 1.0d, 0.0d},
                        new double[]{1.0d, 0.0d, 1.0d},
                        new double[]{1.0d, 1.0d, 1.0d},
                        new double[]{0.0d, 0.0d, 1.0d},
                };
        double[][] outputs =
                {
                        new double[]{0.5d},
                        new double[]{1.0d},
                        new double[]{0.0d},
                        new double[]{0.3d},
                        new double[]{0.7d}
                };
        ActivationFunction activationFunction = new Sigmoid();
        LayerNetwork network = new LayerNetwork(new int[]{2, 1},
                new int[]{3, 2},
                new ActivationFunction[]{activationFunction, activationFunction});
        network.train(5000, 0.1, inputs, outputs);
        for (int i = 0; i < inputs.length; i++) {
            network.setInputLayer(inputs[i]);
            network.compute();
            System.out.println("Output: " + network.getOutput()[0] + " Expected: " + outputs[i][0]);
        }
    }
}
