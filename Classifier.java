import java.util.List;

public class Classifier {

  private int numFeatures;
  private double[] weights;
  private double bias;

  public Classifier(int numFeatures) {
    this.numFeatures = numFeatures;
    this.weights = new double[numFeatures];
    this.bias = 0;
  }

  public void train(List<DataPoint> data) {
    
    // Train using stochastic gradient descent
    for (int i = 0; i < 1000; i++) {
      DataPoint point = getRandomPoint(data);
      double error = getError(point);
      updateWeights(point, error); 
    }
  }

  private double getError(DataPoint point) {
    // Calculate predicted class 
    double prediction = predict(point.features);
    
    // Compare prediction to actual class
    return prediction - point.label; 
  }

  private void updateWeights(DataPoint point, double error) {

    // Update bias
    bias -= error * 0.01;

    // Update weights
    for (int i = 0; i < numFeatures; i++) {
      weights[i] -= error * point.features[i] * 0.01; 
    }
  }

  public double predict(double[] features) {
   
    // Calculate weighted sum 
    double weightedSum = 0;
    for (int i = 0; i < numFeatures; i++) {
      weightedSum += features[i] * weights[i];
    }

    // Add bias 
    weightedSum += bias;

    // Pass through activation function 
    return sigmoid(weightedSum);
  }

  private double sigmoid(double x) {
    return 1 / (1 + Math.exp(-x));
  }

  private DataPoint getRandomPoint(List<DataPoint> data) {
    // Pick random data point for stochastic gradient descent
    return data.get((int)(Math.random() * data.size())); 
  }

}

class DataPoint {
  double[] features;
  double label; 
}