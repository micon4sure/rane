import Neuron from './Neuron';
import Network from './Network'
import Connection from './Connection'
import * as _ from 'lodash'

class Trainer {
  static train(network: Network, data, iterations) {
    console.log(network.export().genome.connections)
    for (let i = 0; i < iterations; i++) {
      const errors = new Array(network.getConfig().output).fill(0);

      //TODO: batches
      _.each(data, example => {
        const result = network.activate(example.input);
        _.each(result, (value, index) => {
          errors[index] += example.output - value;
        });
      })

      if (i % 1000 == 0) {
        console.log('I', i, errors)
      }

      _.each(network.getOutputNeurons(), (neuron: Neuron, index) => {
        Trainer.adjust(neuron, errors[index], network.getConfig().learningRate, .9);
      });

      console.log(network.export().genome.connections)
    }
  }
  static adjust(neuron: Neuron, error, learningRate, multiplier) {
    _.each(neuron.getConnectionsBackward(), (connection: Connection) => {
      const delta = connection.from.getActivation() * error * learningRate * multiplier;
      connection.weight += delta;
      Trainer.adjust(connection.from, error, learningRate, multiplier * multiplier)
    });
  }
}

export default Trainer