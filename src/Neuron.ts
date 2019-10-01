import * as _ from 'lodash'
import Squash from './Squash'
import Connection from './Connection'

export enum NEURON_TYPE {
  input = 'input',
  hidden = 'hidden',
  output = 'output'
}

class Neuron {

  private id: string;
  private type: NEURON_TYPE;
  private bias: number;
  private squash: Function

  private state: number;

  private connectionsForward = new Array<Connection>();
  private connectionsBackward = new Array<Connection>();

  private activations = 0;

  constructor(id: string, type: NEURON_TYPE, bias: number = null, squash: string = 'logistic') {
    this.id = id;
    this.type = type;
    this.squash = Squash[squash];
    this.bias = bias === null ? Math.random() * 2 - 1 : bias;
    this.state = 0;
  }

  connectForward(connection: Connection) {
    this.connectionsForward.push(connection)
  }
  connectBackward(connection: Connection) {
    this.connectionsBackward.push(connection)
  }

  activate(activation: number) {
    // if there has been no activations in this forward pass, the activation passed here is the initial state
    if (this.activations == 0) {
      this.state = activation;
    } else {
      // otherwise, add the activation to the state
      this.state += activation;
    }

    // if all incoming neuron connections have fired
    if (++this.activations >= this.connectionsForward.length) {
      // reset the activations counter
      this.activations = 0;

      // calculate the activation value (squash state + bias)
      const activation = this.getActivation();
      // fire on all outgoing connections
      _.each(this.connectionsForward, connection => {
        connection.to.activate(activation * connection.weight);
      })
    }
  }

  getId() { return this.id; }
  getType() { return this.type; }
  getBias() { return this.bias; }
  getSquash() { return this.squash; }
  
  getState() { return this.state; }
  getActivation() { return this.squash(this.state + this.bias); }

  getConnectionsForward() { return this.connectionsForward; }
  getConnectionsBackward() { return this.connectionsBackward; }

  toJSON() {
    return {
      id: this.id,
      type: this.type,
      bias: this.bias,
      squash: this.squash,
      connections: _.map(this.connectionsForward, connection => {
        return {
          id: connection.to.id,
          weight: connection.weight
        }
      })
    }
  }
}

export default Neuron;