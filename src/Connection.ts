import Neuron from './Neuron'

class Connection {

  public from: Neuron;
  public to: Neuron;
  public weight: number;
  public innovation: number;

  public adjustment: number;

  constructor(from: Neuron, to: Neuron, weight:number, innovation: number) {
    this.from = from;
    this.to = to;
    this.weight = weight;
    this.innovation = innovation;
  }

  toJSON() {
    return {
      innovation: this.innovation,
      from: this.from.getId(),
      to: this.to.getId(),
      weight: this.weight,
      adjustment: this.adjustment,
      enabled: true
    }
  }
}

export default Connection;