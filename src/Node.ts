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

	private pdError_Output_Connected_Sum = 0;
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
  
	/*
    ∂Eₜₒₜₐₗ / ∂wᵢⱼ  = ( ∂Eₜₒₜₐₗ / ∂outⱼ ) * ( ∂outⱼ / ∂netⱼ ) * (  ∂netⱼ / ∂wᵢⱼ )
    
    ∂Eₜₒₜₐₗ / ∂outⱼ = - ( target - outⱼ)
    ∂outⱼ / ∂netⱼ = outⱼ (1 - outⱼ) (derivative of activation function, here: sigmoid)
    ∂netⱼ / ∂wᵢⱼ  = outᵢ

    Δw = -η * ∂Eₜₒₜₐₗ / ∂wᵢⱼ = -η * ∂ⱼ * outᵢ
  */
	propagateOutput(ideal, learningRate) {
		// calculate partial derivatives
		const pdError_Output = ideal - this.getActivation();
    const pdOutput_Net = this.getActivation(true);
    const pdError_Net = pdError_Output * pdOutput_Net;
    //console.log('setting pderrout', {id: this.id, pdError_Out})
    
    //this.adjustment = this.delta = learningRate * pdOutput_Net * pdError_Output;
    //const pdNet_Bias = 1;
    //this.adjustment = pdError_Output * pdNet_Bias;
    
		// for all incoming connections
		_.each(this.connectionsBackward, (connection: Connection) => {
      const pdNet_Input = connection.from.getActivation();
      
			// calculate partial derivative for error to weight of connection
			const pdError_Weight = pdError_Net * pdNet_Input

			// assign weight adjustment to connection
      connection.adjustment = connection.delta = learningRate * pdError_Weight;
      //console.log('adjustment', {from: connection.from.id, to: connection.to.id, pdError_Weight})
  
			// propagate error backwards
      if (connection.from.type == NODE_TYPE.input) return;
      //console.log('UPDATE WEIGHT', {id: connection.innovation, from: connection.from.id, to: connection.to.id, pdError_Out, pdError_Weight, pdNet_Input, pdOutput_Net})
      //console.log('PROPAGATING', {from: connection.from.id, pdError_Net, to: connection.to.id, weight: connection.weight, combined: pdError_Net * connection.weight})
			connection.from.propagateHidden(pdError_Net * connection.weight, learningRate);
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
	propagateHidden(pdError_Output_Connected, learningRate) {
    // add up all incoming error derivatives
    //console.log('adding pdErrOutSum', this.id, this.propagations, pdError_Output_Connected)
    this.pdError_Output_Connected_Sum += pdError_Output_Connected;
    const pdOutput_Net = this.getActivation(true);

		// if all incoming connections have backpropagated
		if (++this.propagations == this.connectionsForward.length) {
      const pdError_Net = this.pdError_Output_Connected_Sum * pdOutput_Net;
      this.propagations = 0;
      
			// assign bias adjustment to node
      //this.adjustment = this.delta = learningRate * pdOutput_Net * this.pdError_Output_Connected_Sum;

			// for all incoming connections
			_.each(this.connectionsBackward, (connection: Connection) => {
        const pdNet_Input = connection.from.getActivation();

				// calculate partial derivative for error to weight of connection
        const pdError_Weight = pdError_Net * pdNet_Input;
				// assign weight adjustment to connection
				connection.adjustment = connection.delta = learningRate * pdError_Weight;
        //console.log('UPDATE WEIGHT', {from: connection.from.id, to: connection.to.id, pdError_Out: this.pdError_Output_Connected_Sum, pdError_Weight, pdNet_Input, pdOutput_Net})

				// propagate errors backwards
        if (connection.from.type == NODE_TYPE.input) return;
				connection.from.propagateHidden(
					pdError_Net * connection.weight,
					learningRate
				);
      });
      this.pdError_Output_Connected_Sum = 0;
		}
	}

	adjust() {
		// actually adjust bias and connection weights (recursively)
		this.bias -= this.adjustment;
		this.adjustment = 0;
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
