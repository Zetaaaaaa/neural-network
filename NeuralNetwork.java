import java.util.Random;

public class NeuralNetwork{

    private double w1, w2, w3, w4;
    private double w5, w6;
    private double b1, b2, b3;
    private double learningRate;

    public NeuralNetwork(double learningRate){
        this.learningRate = learningRate;
        Random rand = new Random();
        w1 = rand.nextGaussian();
        w2 = rand.nextGaussian();
        w3 = rand.nextGaussian();
        w4 = rand.nextGaussian();
        w5 = rand.nextGaussian();
        w6 = rand.nextGaussian();
        b1 = rand.nextGaussian();
        b2 = rand.nextGaussian();
        b3 = rand.nextGaussian();
    }

    private double sigmoid(double x){
        return 1.0/(1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double sigmoidOutput){
        return sigmoidOutput * (1- sigmoidOutput);
    }

    
    public double[] forward(double x1, double x2) {
        double h1 = sigmoid(x1 * w1 + x2 * w2 + b1);
        double h2 = sigmoid(x1 * w3 + x2 * w4 + b2);
        double output = sigmoid(h1 * w5 + h2 * w6 + b3);
        return new double[]{h1, h2, output};
    }

    public void train(double[][] inputs, double[] targets, int epochs){
        for(int epoch = 0; epoch < epochs; epoch++){
            double totalLoss = 0.0;
            for(int i = 0; i < inputs.length; i++){
                double x1 = inputs[i][0];
                double x2 = inputs[i][1];
                double target = targets[i];

                double h1 = sigmoid(x1 * w1 + x2 * w2 + b1);
                double h2 = sigmoid(x1 * w3 + x2 * w4 + b2);
                double output = sigmoid(h1 * w5 + h2 * w6 + b3);

                double error = target - output;
                totalLoss += error * error;
                double deltaOutput = error * sigmoidDerivative(output);

                double deltaH1 = deltaOutput * w5 * sigmoidDerivative(h1);
                double deltaH2 = deltaOutput * w6 * sigmoidDerivative(h2);
                w5 += learningRate * deltaOutput * h1;
                w6 += learningRate * deltaOutput * h2;
                b3 += learningRate * deltaOutput;

                w1 += learningRate * deltaH1 * x1;
                w2 += learningRate * deltaH1 * x2;
                b1 += learningRate * deltaH1;

                w3 += learningRate * deltaH2 * x1;
                w4 += learningRate * deltaH2 * x2;
                b2 += learningRate * deltaH2;
            }
            if (epoch % 1000 == 0) {
                double mse = totalLoss / inputs.length;
                System.out.printf("Epoch %5d - MSE: %.6f%n", epoch, mse);
            }
        }
    }


        public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2.0);

        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        double[] targets = {0, 1, 1, 0};

        System.out.println("Training neural network on XOR...");
        System.out.println("Architecture: 2 inputs -> 2 hidden -> 1 output");
        System.out.println("Activation: Sigmoid | Loss: MSE | Learning Rate: 2.0");
        System.out.println("-------------------------------------------");

        nn.train(inputs, targets, 10000);

        System.out.println("-------------------------------------------");
        System.out.println("Results after training:");
        for (int i = 0; i < inputs.length; i++) {
            double[] result = nn.forward(inputs[i][0], inputs[i][1]);
            double output = result[2];
            System.out.printf("Input: [%.0f, %.0f] -> Output: %.4f (Expected: %.0f)%n",
                inputs[i][0], inputs[i][1], output, targets[i]);
        }
    }
}