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
        let result = network.activate(example.input);
        console.log('RESULT', example, result)
        _.each(result, (value, index) => {
          Trainer.adjust(network.getOutputNeurons()[index], example.output, network.getConfig().learningRate)
        });
        result = network.activate(example.input);
        console.log('ADJUSTED', example, result)
      })
    }
  }

  static adjust(neuron: Neuron, target, learningRate, errorPrev = null) {
    let error;
    _.each(neuron.getConnectionsBackward(), connection => {
      if (neuron.getType() == NEURON_TYPE.output) {
        error = neuron.getActivation(true) * (neuron.getActivation(true) - target)
      } else {
        error = neuron.getActivation(true) * (errorPrev * connection.weight);
      }
      const deltaWeight = -learningRate * error * neuron.getActivation();
      connection.weight += deltaWeight;
      Trainer.adjust(connection.from, target, learningRate, error)
    })
  }
}

export default Trainer