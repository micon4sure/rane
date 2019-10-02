import Neuron, { NEURON_TYPE } from './Neuron';
import Network from './Network'
import Connection from './Connection'
import * as _ from 'lodash'
import squash from './squash'
import { generateKeyPair } from 'crypto';
import { nodeInternals } from 'stack-utils';

class Trainer {

  /**
   * Train the network on some data, return the error sum after training
   * @param network 
   * @param data 
   * @param iterations 
   */
  static train(network: Network, data, iterations) {
    //console.log(network.export().genome.connections)
    for (let i = 0; i < iterations; i++) {
      let errors = new Array(network.getConfig().output).fill(0);
      _.each(data, example => {
        const result = network.activate(example.input);
        //console.log('RESULT', example.input, example.output, result)
        _.each(result, (value, index) => {
          Trainer.adjust(network.getOutputNeurons()[index], example.output, network.getConfig().learningRate)
        });
      })
    }
  }

  static adjust(neuron: Neuron, target, learningRate, deltaPrev = null) {
    let delta;
    _.each(neuron.getConnectionsBackward(), connection => {
      if (neuron.getType() == NEURON_TYPE.hidden) {
        delta = neuron.getActivation(true) * (neuron.getActivation(true) - target)
      } else {
        delta = neuron.getActivation(true) * (deltaPrev * connection.weight);
      }
      const deltaWeight = -learningRate * delta * neuron.getActivation();
      connection.weight += deltaWeight;
      //Trainer.adjust(connection.from, target, learningRate, delta)
    })
  }
}

export default Trainer