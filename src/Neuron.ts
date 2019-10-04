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

  private partialDerivativeErrorOutConnectedSum = 0;
  private state: number;

  private connectionsForward = new Array<Connection>();
  private connectionsBackward = new Array<Connection>();

  private activations = 0;
  private activationValues = {};
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

  activate(activation: number, memory: Memory, connection: Connection = null) {
    if(connection !== null)
      this.activationValues[connection.innovation] = activation;
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
        connection.to.activate(activation * connection.weight, memory, connection);
        memory.activated(connection.innovation);
      })
    }
  }

  /*
    ∂Eₜₒₜₐₗ / ∂wᵢⱼ  = ( ∂Eₜₒₜₐₗ / ∂outⱼ ) * ( ∂outⱼ / ∂netⱼ ) * (  ∂netⱼ / ∂wᵢⱼ )
    
    ∂Eₜₒₜₐₗ / ∂outⱼ = - ( target - outⱼ)
    ∂outⱼ / ∂netⱼ = outⱼ (1 - outⱼ)
    ∂netⱼ / ∂wᵢⱼ  = outᵢ

    Δw = -η * ∂Eₜₒₜₐₗ / ∂wᵢⱼ
  */
  propagateOutput(ideal, learningRate) {
    const partialDerivativeErrorOut = -(ideal - this.getActivation())
    const partialDerivativeOutNetinput = this.getActivation() * (1 - this.getActivation());

    _.each(this.connectionsBackward, (connection: Connection) => {
      const partialDerivativeNetinputWeight = connection.from.getActivation()

      const partialDerivativeErrorWeight = partialDerivativeErrorOut * partialDerivativeOutNetinput * partialDerivativeNetinputWeight;
      connection.adjustment = -learningRate * partialDerivativeErrorWeight;
      
      connection.from.propagateHidden(partialDerivativeErrorOut, connection.weight, learningRate);
    });
  }

  /*
    ∂Eₜₒₜₐₗ / ∂wᵢⱼ  = ( ∂Eₜₒₜₐₗ / ∂outⱼ ) * ( ∂outⱼ / ∂netⱼ ) * (  ∂netⱼ / ∂wᵢⱼ )

    ∂Eₜₒₜₐₗ / ∂outⱼ = Σ [ (∂Eₖ₁ / doutⱼ) + (∂Eₖ₂ / doutⱼ) + ... (∂Eₖₙ / / doutⱼ) ]
                  -> ∂Eₖ / ∂outⱼ   = ∂Eₖ * / ∂netₖ
                                                -> ∂netₖ / ∂outⱼ = wᵢⱼ
    ∂outⱼ / ∂netⱼ = outⱼ (1 - outⱼ)
    ∂netⱼ / ∂wᵢⱼ  = input from this connection

    Δw = -η * ∂Eₜₒₜₐₗ / ∂wᵢⱼ
  */
  propagateHidden(partialDerivativeErrorOutConnected, weight, learningRate) {
    this.partialDerivativeErrorOutConnectedSum += partialDerivativeErrorOutConnected;
    if(++this.propagations == this.connectionsForward.length) {
      this.propagations = 0;
      const partialDerivativeOutNetinput = this.getActivation() * (1 - this.getActivation());
      _.each(this.connectionsBackward, (connection: Connection) => {
        const partialDerivativeNetinputWeight = this.activationValues[connection.innovation];

        const partialDerivativeErrorWeight = this.partialDerivativeErrorOutConnectedSum * partialDerivativeOutNetinput * partialDerivativeNetinputWeight;
        connection.adjustment = -learningRate * partialDerivativeErrorWeight;
          
        connection.from.propagateHidden(this.partialDerivativeErrorOutConnectedSum, connection.weight, learningRate)
      });
    }
  }

  adjust(memory) {
    _.each(this.getConnectionsBackward(), (connection: Connection) => {
      if (!memory.allowed(connection.innovation)) return;
      connection.weight += connection.adjustment;
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
      activations: this.activations,
      propagations: this.propagations,
      enabled: true,
      connections: _.map(this.connectionsForward, connection => {
        return {
          to: connection.to.id,
          weight: connection.weight
        }
      })
    }
  }
}

export default Neuron;