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

	private partialDerivativeErrorOutConnectedSum = 0;
	private netInput: number;

	private adjustment: number = 0;
	private delta: number = 0;

	private connectionsForward = new Array<Connection>();
	private connectionsBackward = new Array<Connection>();

  public activation;
	private activations = 0;
  private propagations = 0;
  
  private activationValues = {};

  public pdError_Output = 0;
  public pdError_Net = 0;

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

    this.activation = this.getActivation(true);
    
    if(connection !== null)
      this.activationValues[connection.innovation] = activation;

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

	adjust(learningRate) {
    const pdOutput_Net  = this.getActivation(true);
    this.pdError_Net = this.pdError_Output * pdOutput_Net;
    _.each(this.connectionsBackward, connection => {
      const pdNet_Input = connection.from.getActivation();
      const pdError_Weight = this.pdError_Net * pdNet_Input
      console.log('adjusting!', {node: this.id, from: connection.from.id, w: connection.innovation, inputs: this.activationValues, pdErrorOut: this.pdError_Output, pdNet_Input,pdOutput_Net, pdError_Weight})
      connection.adjustment = learningRate * pdError_Weight;
    });
	}

  getId() { return this.id; }
  getType() { return this.type; }
  getBias() { return this.bias; }
  getSquash() { return this.squash; }

  getState() { return this.netInput; }
  getActivation(derivative = false) {
    if(this.type == NODE_TYPE.input) {
      return this.netInput;
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
