import Neuron from './Neuron'

class Connection {

  public from: Neuron;
  public to: Neuron;
  public weight: number;
  public innovation: number;

  constructor(from: Neuron, to: Neuron, weight:number, innovation: number) {
    this.from = from;
    this.to = to;
    this.weight = weight;
    this.innovation = innovation;
  }
}

export default Connection;