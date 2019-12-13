import * as _ from "lodash";
import Squash from "./Squash";
import Connection from "./Connection";

export enum NODE_TYPE {
  input = "input",
  hidden = "hidden",
  output = "output"
}

class Node {
  private id: number;
  private type: NODE_TYPE;
  private squash: Function;

  private signalErrorSum: number = 0;
  private bias: number = 1;
  private netInput: number = 0;
  private output: number = 0;

  private adjustment: number = 0;
  private adjustmentModifier: number = 0;
  private delta: number = 0;

  private connectionsForward = new Array<Connection>();
  private connectionsBackward = new Array<Connection>();

  private activations = 0;
  private propagations = 0;

  constructor(
    id: number,
    type: NODE_TYPE,
    bias: number = null,
    squash: string = "sigmoid"
  ) {
    this.id = id;
    this.type = type;
    this.squash = Squash[squash];
    this.bias = bias;
  }

  connectForward(connection: Connection) {
    this.connectionsForward.push(connection);
  }
  connectBackward(connection: Connection) {
    this.connectionsBackward.push(connection);
  }

  activate(activation: number, connection: Connection = null) {
    // if there has been no activations in this forward pass, the activation passed here is the initial netInput
    if (this.activations == 0) {
      this.netInput = activation;
    } else {
      // otherwise, add the activation to the netInput
      this.netInput += activation;
    }

    // if all incoming node connections have fired
    if (++this.activations >= this.connectionsBackward.length) {
      // reset the activations counter
      this.activations = 0;

      // calculate the activation value (squash net input)
      // except if input node (then it's just unchanged input value)
      if (this.type == NODE_TYPE.input) {
        this.output = this.netInput;
      } else {
        this.netInput += this.bias;
        this.output = this.squash(this.netInput, false);
      }

      // fire on all outgoing connections
      _.each(this.connectionsForward, (connection: Connection) => {
        connection.to.activate(this.output * connection.weight, connection);
      });
    }
  }

  propagateOutput(ideal, learningRate, momentum) {
    // calculate signal error (partial derivative of activation with respect to net input)
    const signalError = this.squash(this.netInput, true) * (this.output - ideal);

    // set bias delta and modifier
    this.adjustment = -learningRate * signalError * this.squash(this.netInput, true);
    this.adjustmentModifier = this.adjustment * momentum;

    _.each(this.connectionsBackward, connection => {
      // set weight delta and modifier
      connection.adjustment = connection.delta = -learningRate * signalError * connection.from.output;
      connection.adjustmentModifier = connection.adjustment * momentum;

      // propagate backwards
      connection.from.propagateHidden(signalError * connection.weight, learningRate, momentum);
    });
  }

  propagateHidden(signalError, learningRate, momentum) {
    // sum up incoming signal error for later use
    this.signalErrorSum += signalError;

    // note: input nodes won't ever hit this condition as is intended
    if (++this.propagations == this.connectForward.length) {
      // all incoming connections have fired (reset counter)
      this.propagations = 0;

      // set bias delta and modifier
      this.adjustment = -learningRate * this.signalErrorSum * this.squash(this.netInput, true);
      this.adjustmentModifier = this.adjustment * momentum;

      // calculate signal error (partial derivative of activation with respect to net input)
      const signalError = this.squash(this.netInput, true) * this.signalErrorSum;
      _.each(this.connectionsBackward, connection => {
        // set weight delta and modifier
        connection.adjustment = connection.delta = -learningRate * signalError * connection.from.output;
        connection.adjustmentModifier = connection.adjustment * momentum;

        // propagate backwards
        connection.from.propagateHidden(signalError, learningRate, momentum);
      })
    }
  }

  adjust() {
    this.bias += this.adjustment + this.adjustmentModifier;
    this.adjustment = 0;
    this.signalErrorSum = 0;
  }

  getId() { return this.id; }
  getType() { return this.type; }

  getNetInput() { return this.netInput; }
  getOutput() { return this.output; }

  getBias() { return this.bias; }

  getConnectionsForward() { return this.connectionsForward; }
  getConnectionsBackward() { return this.connectionsBackward; }

  toJSON() {
    return {
      id: this.id,
      type: this.type,
      bias: this.bias,
      squash: this.squash.name,
      output: this.output,
      netInput: this.netInput,
      activations: this.activations,
      propagations: this.propagations,
      enabled: true,
      delta: this.delta,
      connections: {
        outgoing: _.map(this.connectionsForward, connection => {
          return {
            to: connection.to.id,
            weight: connection.weight,
            delta: connection.delta
          };
        }),
        incoming: _.map(this.connectionsBackward, connection => {
          return {
            to: connection.to.id,
            weight: connection.weight,
            delta: connection.delta
          };
        })
      }
    };
  }
}

export default Node;