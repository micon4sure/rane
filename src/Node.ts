import * as _ from "lodash";
import Squash from "./Squash";
import Connection from "./Connection";
import Memory from "./Memory";

export enum NODE_TYPE {
  input = "input",
  hidden = "hidden",
  output = "output"
}

class Node {
  private id: number;
  private type: NODE_TYPE;
  private bias: number;
  private squash: Function;

  private sum_dk_wjk = 0;
  private netInput: number;

  private adjustment: number = 0;
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
    this.netInput = 0;
  }

  connectForward(connection: Connection) {
    this.connectionsForward.push(connection);
  }
  connectBackward(connection: Connection) {
    this.connectionsBackward.push(connection);
  }

  activate(activation: number, memory: Memory, connection: Connection = null) {
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

      // calculate the activation value (squash state + bias)
      // except if input node (then it's just unchanged input value)
      const output = this.getActivation();
      // fire on all outgoing connections
      _.each(this.connectionsForward, (connection: Connection) => {
        if (!memory.allowed(connection.innovation)) return;
        connection.to.activate(output * connection.weight, memory, connection);
        memory.activated(connection.innovation);
      });
    }
  }

  /**
   * ∂Cost/weight = ∂Cost / ∂A * ∂A / ∂Z * ∂Z / ∂W
   * ∂Cost/bias = ∂Cost / ∂A * ∂A / ∂Z * ∂Z / ∂B
   * ∂Cost/A-1 = ∂Cost / ∂A * ∂A / ∂Z * ∂Z / ∂B
   * ∂Z / ∂B = 1
   */
  /**
   * (pdError_Weight = pdError_Net * pdNet_Input)
   * 
   * dj == pdError_Net (pdError_Output * pdOutput_Net)
   * pdNet_Input = incoming value from connection
   */
   propagateOutput(ideal, learningRate) {
    const dj = this.getActivation(true) * (this.getActivation() - ideal)
    _.each(this.connectionsBackward, connection => {
      connection.adjustment = connection.delta =  -learningRate * dj * connection.from.getActivation();
      connection.from.propagateHidden(connection.weight * dj, learningRate)
    });
  }

  /**
   * (pdError_Weight = pdError_Net * pdNet_Input)
   * 
   * dj == pdError_Net (pdError_Output * pdOutput_Net)
   * pdNet_Input = incoming value from connection
   */
  propagateHidden(dk_wjk, learningRate) {
    this.sum_dk_wjk += dk_wjk;
    if (++this.propagations >= this.connectionsForward.length) {
      const dj = this.getActivation(true) * this.sum_dk_wjk
      
      _.each(this.connectionsBackward, connection => {
        connection.adjustment = connection.delta = -learningRate * dj * connection.from.getActivation();
        
        connection.from.propagateHidden(connection.weight * dj, learningRate)
      });
    }
  }

  adjust() {
    //this.bias += this.adjustment;
    this.sum_dk_wjk = 0;
  }

  getId() { return this.id; }
  getType() { return this.type; }
  getBias() { return this.bias; }
  getSquash() { return this.squash; }

  getUnsquished() {
    return this.netInput + this.bias;
  }
  getActivation(derivative = false) {
    if (this.type == NODE_TYPE.input) {
      return this.netInput;
    }
    if(derivative) {
      //return this.squash(this.getActivation(), true)
    }
    return this.squash(this.netInput + this.bias, derivative);
  }
  getNetinput() { return this.netInput; }

  getConnectionsForward() { return this.connectionsForward; }
  getConnectionsBackward() { return this.connectionsBackward; }

  toJSON() {
    return {
      id: this.id,
      type: this.type,
      bias: this.bias,
      squash: this.squash.name,
      activation: this.getActivation(),
      netInput: this.getNetinput(),
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