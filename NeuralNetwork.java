import java.util.Random;

public class NeuralNetwork{

    private double w1, w2, w3, w4;
    private double w5, w6;
    private double b1, b2, b3;
    private double learningRate;

    public NeuralNetwork(double learningRate){
        learningRate = this.learningRate;
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
        return 1.0/(1.0 + Math.exp(-1));
    }

    private double sigmoidDerivative(double sigmoidOutput){
        return sigmoidOutput * (1- sigmoidOutput);
    }

    public static void main(String args[]){
        System.out.println("Hello from Neural Network");
    }
}