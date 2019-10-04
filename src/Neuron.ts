import * as _ from 'lodash'
import Squash from './Squash'
import Connection from './Connection'
import Memory from './Memory'

export enum NEURON_TYPE {
  input = 'input',
  hidden = 'hidden',
  output = 'output'
}

class Neuron {

  private id: number;
  private type: NEURON_TYPE;
  private bias: number;
  private squash: Function

  private error: number = 0;
  private state: number;

  private connectionsForward = new Array<Connection>();
  private connectionsBackward = new Array<Connection>();

  private activations = 0;
  private propagations = 0;

  constructor(id: number, type: NEURON_TYPE, bias: number = null, squash: string = 'sigmoid') {
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

  activate(activation: number, memory: Memory) {
    // if there has been no activations in this forward pass, the activation passed here is the initial state
    if (this.activations == 0) {
      this.state = activation;
    } else {
      // otherwise, add the activation to the state
      this.state += activation;
    }

    // if all incoming neuron connections have fired
    if (++this.activations >= this.connectionsBackward.length) {
      // reset the activations counter
      this.activations = 0;

      // calculate the activation value (squash state + bias)
      // but don't squash or apply bias on input neurons
      activation = this.type == NEURON_TYPE.input ? activation : this.getActivation();
      // fire on all outgoing connections
      _.each(this.connectionsForward, (connection: Connection) => {
        if (!memory.allowed(connection.innovation)) return;
        connection.to.activate(activation * connection.weight, memory);
        memory.activated(connection.innovation);
      })
    }

  }

  propagate(error, memory: Memory) {
    this.propagations++;
    this.error += error;

    // derivative of output to input
    const derivativeOutputInput = this.getActivation(true)

    if (this.propagations >= this.connectionsForward.length) {
      this.propagations = 0;
      _.each(this.getConnectionsBackward(), (connection: Connection) => {
        if (!memory.allowed(connection.innovation)) return;
          // derivative of input to weight
          const derivativeInputWeight = connection.to.getType() == 'output'
           ? connection.to.getActivation()
           : connection.to.getState();

          // derivative of ideal_output error to delta weight
          const derivativeErrorWeight = this.error * derivativeOutputInput * derivativeInputWeight;

          // calculate the delta for the connection
          connection.adjustment = derivativeErrorWeight;

          if(connection.innovation == 1)console.log({
            id: this.id,
            connection: connection.innovation,
            delta: connection.adjustment,
            type: connection.from.type,
            'derivative error/output': [error, this.error],
            'derivative output/input': derivativeOutputInput,
            'derivative input/weight': derivativeInputWeight,
            'derivative error/weight': derivativeErrorWeight,
          })

          // propagate the error
          connection.from.propagate(this.error * derivativeOutputInput * connection.weight, memory);
          memory.activated(connection.innovation)
        });
    }
  }

  adjust(memory) {
    _.each(this.getConnectionsBackward(), (connection: Connection) => {
      if(!memory.allowed(connection.innovation)) return;
      const delta = - .5 * connection.adjustment
      if(connection.innovation == -1)console.log('ADJUSTING', {
        innovation: connection.innovation,
        adjustment: connection.adjustment, 
        weight: connection.weight,
        delta,
        result: connection.weight + delta
      })
      connection.weight += delta;
      connection.adjustment = 0;
      memory.activated(connection.innovation);
      connection.from.adjust(memory)
    });
  }

  getId() { return this.id; }
  getType() { return this.type; }
  getBias() { return this.bias; }
  getSquash() { return this.squash; }

  getState() { return this.state; }
  getActivation(derivative = false) { return this.squash(this.state + this.bias, derivative); }

  getConnectionsForward() { return this.connectionsForward; }
  getConnectionsBackward() { return this.connectionsBackward; }

  setBias(bias) { this.bias = bias; }

  toJSON() {
    return {
      id: this.id,
      type: this.type,
      bias: this.bias,
      squash: this.squash,
      activation: this.getActivation(),
      state: this.getState(),
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