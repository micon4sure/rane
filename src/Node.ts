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

	private activations = 0;
	private activationValues = {};
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
		if (connection !== null)
			this.activationValues[connection.innovation] = activation;
		// if there has been no activations in this forward pass, the activation passed here is the initial state
		if (this.activations == 0) {
			this.netInput = activation;
		} else {
			// otherwise, add the activation to the state
			this.netInput += activation;
		}

		// if all incoming node connections have fired
		if (++this.activations >= this.connectionsBackward.length) {
			// reset the activations counter
			this.activations = 0;

			// calculate the activation value (squash state + bias)
			// except if input node
      const output = this.getActivation();
			// fire on all outgoing connections
			_.each(this.connectionsForward, (connection: Connection) => {
				if (!memory.allowed(connection.innovation)) return;
				connection.to.activate(output * connection.weight, memory, connection);
				memory.activated(connection.innovation);
			});
		}
	}

	/*
    ∂Eₜₒₜₐₗ / ∂wᵢⱼ  = ( ∂Eₜₒₜₐₗ / ∂outⱼ ) * ( ∂outⱼ / ∂netⱼ ) * (  ∂netⱼ / ∂wᵢⱼ )
    
    ∂Eₜₒₜₐₗ / ∂outⱼ = - ( target - outⱼ)
    ∂outⱼ / ∂netⱼ = outⱼ (1 - outⱼ) (derivative of activation function, here: sigmoid)
    ∂netⱼ / ∂wᵢⱼ  = outᵢ

    Δw = -η * ∂Eₜₒₜₐₗ / ∂wᵢⱼ = -η * ∂ⱼ * outᵢ
  */
	propagateOutput(ideal, learningRate) {
		// calculate partial derivatives
		const partialDerivativeErrorOut = -(ideal - this.getActivation());
    const partialDerivativeOutNetinput = this.getActivation(true);
    
		this.adjustment = this.delta =
    learningRate * partialDerivativeOutNetinput * partialDerivativeErrorOut;
    
		// for all incoming connections
		_.each(this.connectionsBackward, (connection: Connection) => {
      const partialDerivativeNetinputWeight = connection.from.getActivation();
      const partialDerivativeErrorOutNetinput = partialDerivativeErrorOut * partialDerivativeOutNetinput;
      
			// calculate partial derivative for error to weight of connection
			const derivativeErrorWeight = partialDerivativeErrorOut * partialDerivativeOutNetinput * partialDerivativeNetinputWeight;

			// assign weight adjustment to connection
			connection.adjustment = connection.delta = learningRate * derivativeErrorWeight;
  
			// propagate error backwards
			if (connection.from.type == NODE_TYPE.input) return;
			connection.from.propagateHidden(partialDerivativeErrorOutNetinput * connection.weight, learningRate);
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
	propagateHidden(partialDerivativeErrorOutConnected, learningRate) {
		// add up all incoming error derivatives
    this.partialDerivativeErrorOutConnectedSum += partialDerivativeErrorOutConnected;

		// if all incoming connections have backpropagated
		if (++this.propagations == this.connectionsForward.length) {
			this.propagations = 0;

			// calculate partial derivatives
			const partialDerivativeOutNetinput = this.getActivation(true);
			const partialDerivativeNetinputBias = 1;

			// calculate partial derivative for error to bias
			const derivativeErrorBias = this.partialDerivativeErrorOutConnectedSum * partialDerivativeOutNetinput * partialDerivativeNetinputBias;

			// assign bias adjustment to node
			this.adjustment = learningRate * derivativeErrorBias;

			// for all incoming connections
			_.each(this.connectionsBackward, (connection: Connection) => {
				const partialDerivativeNetinputWeight = connection.from.getNetinput()

				// calculate partial derivative for error to weight of connection
				const derivativeErrorWeight = this.partialDerivativeErrorOutConnectedSum * partialDerivativeOutNetinput * partialDerivativeNetinputWeight;
				// assign weight adjustment to connection
				connection.adjustment = connection.delta = learningRate * derivativeErrorWeight;

				// propagate errors backwards
				if (connection.from.type == NODE_TYPE.input) return;
				connection.from.propagateHidden(
					this.partialDerivativeErrorOutConnectedSum,
					learningRate
				);
			});
		}
	}

	adjust(memory) {
		// actually adjust bias and connection weights (recursively)
		this.bias -= this.adjustment;
		this.adjustment = 0;
		_.each(this.getConnectionsBackward(), (connection: Connection) => {
			if (!memory.allowed(connection.innovation)) return;
			connection.weight -= connection.adjustment;
			connection.adjustment = 0;
			memory.activated(connection.innovation);
			connection.from.adjust(memory);
		});
	}

	getId() {
		return this.id;
	}
	getType() {
		return this.type;
	}
	getBias() {
		return this.bias;
	}
	getSquash() {
		return this.squash;
	}

	getNetinput() {
		return this.netInput;
	}
	getActivation(derivative = false) {
		if (this.type == NODE_TYPE.input) {
			return this.netInput;
		}
		return this.squash(this.netInput + this.bias, derivative);
	}

	getConnectionsForward() {
		return this.connectionsForward;
	}
	getConnectionsBackward() {
		return this.connectionsBackward;
	}

	setBias(bias) {
		this.bias = bias;
	}

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
			activationValues: this.activationValues,
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
