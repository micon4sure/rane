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

  private pdError_Out_Sum: number = 0;
  private signalErrorSum: number = 0;
  private adjustment = 0;
  private adjustmentModifier = 0;
  private bias: number = 1;
  private netInput: number = 0;
  private output: number = 0;

  private delta: number = 0;

  private connectionsForward = new Array<Connection>();
  private connectionsBackward = new Array<Connection>();

  private activations = 0;
  private propagations = 0;

  private config;

  constructor(id: number, type: NODE_TYPE, bias: number = null, squash: string = "sigmoid", config) {
    this.id = id;
    this.type = type;
    this.squash = Squash[squash];
    this.bias = bias;
    this.config = config;
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
  
  /**
   * let ⱼ = current neuron; ᵢ = prev neuron; ᵗ = current pass; 
   * let η = learning rate; μ = momentum
   * let W = weight; B = Bias
   * 
   * ∂Error/∂Wᵢⱼ = ∂Error/∂Output * ∂Output/∂Net * ∂Net/∂Wᵢⱼ
   * -> ∂Error/∂Output = (Outputⱼ - idealⱼ)
   * -> ∂Output/∂Net = φ'(netⱼ)
   * -> ∂Net/∂Wᵢⱼ = Outputᵢ
   * --> ∂Error/∂Net = ∂Error/∂Output * ∂Output/∂Net
   * --------------------------------------------------------------
   * ∂Error/∂B = ∂Error/∂Output * ∂Output/∂Net * (∂Net/∂B == 1)
   * 
   * ΔWᵢⱼ = (ΔWᵢⱼᵗ⁻¹ * μ) + (-η * ∂Error/∂W)
   * ΔBⱼ = (ΔBⱼᵗ⁻¹ * μ) + (-η * ∂Error/∂B)
   */
  propagateOutput(ideal) {
    // calculate partial derivatives
    const pdError_Output = (this.output - ideal);
    const pdOutput_Net = this.squash(this.netInput, true);
    const pdError_Net = pdOutput_Net * pdError_Output;

    // calculate bias derivative and set bias delta
    const pdError_Bias = pdError_Net * pdOutput_Net;
    //this.delta = this.delta * this.config.momentum;
    this.delta = -this.config.learningRate * pdError_Bias;


    _.each(this.connectionsBackward, connection => {
      // calculate weight derivative and set weight delta
      const pdNet_Weight = connection.from.output;
      connection.delta = connection.delta * this.config.momentum;
      connection.delta += -this.config.learningRate * pdError_Net * pdNet_Weight;

      // propagate backwards
      connection.from.propagateHidden(pdError_Net * connection.weight);
    });
  }

  /**
   * let ⱼ = current neuron; ᵢ = prev neuron; ₖ = next neuron; ᵗ = current pass; 
   * let η = learning rate; μ = momentum
   * let W = weight; B = Bias
   * 
   * ∂Error/∂Wᵢⱼ = ∂Error/∂Output * ∂Output/∂Net * ∂Net/∂Wᵢⱼ
   * -> ∂Error/∂Output = Σₖ [∂Outputₖ/∂Netₖ * Wⱼₖ]
   * -> ∂Output/∂Net = φ'(netⱼ)
   * -> ∂Net/∂Wᵢⱼ = Outputᵢ
   * --> ∂Error/∂Net = ∂Error/∂Output * ∂Output/∂Net
   * --------------------------------------------------------------
   * ∂Error/∂B = ∂Error/∂Output * ∂Output/∂Net * (∂Net/∂B == 1)
   * 
   * ΔWᵢⱼ = (ΔWᵢⱼᵗ⁻¹ * μ) + (-η * ∂Error/∂W)
   * ΔBⱼ = (ΔBⱼᵗ⁻¹ * μ) + (-η * ∂Error/∂B)
   */
  propagateHidden(pdError_Out_Connected) {
    // sum up incoming signal error for later use
    this.pdError_Out_Sum += pdError_Out_Connected;

    // note: input nodes won't ever hit this condition as is intended
    if (++this.propagations == this.connectForward.length) {
      // all incoming connections have fired (reset counter)
      this.propagations = 0;

      // calculate partial derivatives
      const pdOutput_Net = this.squash(this.netInput, true);
      const pdError_Net = pdOutput_Net * this.pdError_Out_Sum;

      // calculate bias derivative and set bias delta
      const pdError_Bias = pdError_Net * pdOutput_Net;
      //this.delta = this.delta * this.config.momentum;
      this.delta = -this.config.learningRate * pdError_Bias;

      _.each(this.connectionsBackward, connection => {
        // calculate weight derivative and set weight delta
        const pdNet_Weight = connection.from.output;
        connection.delta = connection.delta * this.config.momentum;
        connection.delta += -this.config.learningRate * pdError_Net * pdNet_Weight;

        // propagate backwards
        connection.from.propagateHidden(pdError_Net);
      })
    }
  }

  adjust() {
    if(this.getType() != NODE_TYPE.input) {
      //this.bias += this.adjustment < 0 ? 0 : this.adjustment;
      this.bias += this.delta;
      this.delta = 0;
    }

    this.pdError_Out_Sum = 0;
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