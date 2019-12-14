import Node from './Node'

class Connection {

  public from: Node;
  public to: Node;
  public weight: number;
  public innovation: number;

  public delta: number = 0;

  constructor(from: Node, to: Node, weight:number, innovation: number) {
    this.from = from;
    this.to = to;
    this.weight = weight;
    this.innovation = innovation;
  }

  adjust() {
    //this.weight += this.adjustment < 0 ? 0 : this.adjustment;
    this.weight += this.delta;
  }

  toJSON() {
    return {
      innovation: this.innovation,
      from: this.from.getId(),
      to: this.to.getId(),
      weight: this.weight,
      adjustment: this.delta,
      delta: this.delta,
      enabled: true
    }
  }
}

export default Connection;