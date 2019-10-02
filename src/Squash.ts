export default {
  sigmoid(x: number, derivative = false) {
    if(!derivative) return 1 / (1 + Math.exp(-x));
    return Math.exp(-x) / (Math.pow((1 + Math.exp(-x)), 2));
  }
}