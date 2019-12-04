import Node from './Node'

class Connection {

  public from: Node;
  public to: Node;
  public weight: number;
  public innovation: number;
  public plasticity: number;

  public adjustment: number;
  public delta: number;

  constructor(from: Node, to: Node, weight:number, innovation: number, plasticity: number) {
    this.from = from;
    this.to = to;
    this.weight = weight;
    this.innovation = innovation;
    this.plasticity = plasticity;
  }

  toJSON() {
    return {
      innovation: this.innovation,
      plasticity: this.plasticity,
      from: this.from.getId(),
      to: this.to.getId(),
      weight: this.weight,
      adjustment: this.adjustment,
      delta: this.delta,
      enabled: true
    }
  }
}

export default Connection;