export default {
  sigmoid(x: number, derivative = false) {
    if(!derivative) return 1 / (1 + Math.exp(-x));
    return Math.exp(-x) / (Math.pow((1 + Math.exp(-x)), 2));
  },
  relu(x: number, derivative = false) {
    if(!derivative)
      return x > 0 ? x : 0;
    return x > 0 ? 1 : 0;
  }
}