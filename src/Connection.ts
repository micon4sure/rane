import Node from './Node'

class Connection {

  public from: Node;
  public to: Node;
  public weight: number;
  public innovation: number;

  public adjustment: number;
  public delta: number;

  constructor(from: Node, to: Node, weight:number, innovation: number) {
    this.from = from;
    this.to = to;
    this.weight = weight;
    this.innovation = innovation;
  }

  adjust() {
    this.weight += this.adjustment;
    this.adjustment = 0;
  }

  toJSON() {
    return {
      innovation: this.innovation,
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