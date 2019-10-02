import Port from '../src/Port'
import Neuron, { NEURON_TYPE } from '../src/Neuron';
import Network from '../src/Network'
import NEAT from '../src/NEAT'
import * as _ from 'lodash'
import Trainer from '../src/Trainer';
import Genome from '../src/Genome';
import squash from '../src/squash';
import * as fs from 'fs';


test('testNEAT', () => {
  return;
  const neat = new NEAT(12, 4, { input: 2, output: 1, learningRate: 0.001 });
  const AND = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [0] },
    { input: [1, 0], output: [0] },
    { input: [1, 1], output: [1] }

  ]
  return;
  const network = neat.train(AND, 2, 2)

  _.each(AND, example => {
    return;
    const result = network.activate(example.input);
    console.log('AND', example.input, example.output[0], result[0])
    expect(Math.round(result[0])).toEqual(example.output[0]);
  });
  Port.export(network)
});

test('mutateNEAT', () => {
  return;
  const neat = new NEAT(10, 4, { input: 2, output: 1, learningRate: 0.001 });

  const AND = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [0] },
    { input: [1, 0], output: [0] },
    { input: [1, 1], output: [1] }
  ]

  const network = neat.train(AND, 10000, 10);
  _.each(AND, example => {
    const result = network.activate(example.input);
    console.log('AND', example.input, example.output[0], result[0])
  });


  Port.export(network)
});
test('merp', () => {
  const neat = new NEAT(1, 1, { input: 2, output: 1, learningRate: 0.1 });

  const createGenomeWithHidden = () => {
    const genome = new Genome();
    genome.addNodeGene(0, NEURON_TYPE.input, Math.random() * 2 - 1, 'sigmoid', true);
    genome.addNodeGene(1, NEURON_TYPE.input, Math.random() * 2 - 1, 'sigmoid', true);
    genome.addNodeGene(2, NEURON_TYPE.hidden, Math.random() * 2 - 1, 'sigmoid', true);
    genome.addNodeGene(3, NEURON_TYPE.hidden, Math.random() * 2 - 1, 'sigmoid', true);
    genome.addNodeGene(4, NEURON_TYPE.hidden, Math.random() * 2 - 1, 'sigmoid', true);
    genome.addNodeGene(5, NEURON_TYPE.hidden, Math.random() * 2 - 1, 'sigmoid', true);
    genome.addNodeGene(6, NEURON_TYPE.output, Math.random() * 2 - 1, 'sigmoid', true);

    let innovation = 0;
    genome.addConnectionGene(0, 2, Math.random() * 2 - 1, innovation++, true)
    genome.addConnectionGene(1, 2, Math.random() * 2 - 1, innovation++, true)
    genome.addConnectionGene(0, 3, Math.random() * 2 - 1, innovation++, true)
    genome.addConnectionGene(1, 3, Math.random() * 2 - 1, innovation++, true)
    genome.addConnectionGene(2, 4, Math.random() * 2 - 1, innovation++, true)
    genome.addConnectionGene(3, 4, Math.random() * 2 - 1, innovation++, true)
    genome.addConnectionGene(2, 5, Math.random() * 2 - 1, innovation++, true)
    genome.addConnectionGene(3, 5, Math.random() * 2 - 1, innovation++, true)
    genome.addConnectionGene(4, 6, Math.random() * 2 - 1, innovation++, true)
    genome.addConnectionGene(5, 6, Math.random() * 2 - 1, innovation++, true)
    return genome;
  };
  const createGenomeWithoutHidden = () => {
    const genome = new Genome();

    let id = 0;
    genome.addNodeGene(id++, NEURON_TYPE.input, Math.random() * 2 - 1, 'sigmoid', true);
    genome.addNodeGene(id++, NEURON_TYPE.input, Math.random() * 2 - 1, 'sigmoid', true);
    genome.addNodeGene(id++, NEURON_TYPE.output, Math.random() * 2 - 1, 'sigmoid', true);

    let innovation = 0;
    genome.addConnectionGene(0, 2, Math.random() * 2 - 1, innovation++, true)
    genome.addConnectionGene(1, 2, Math.random() * 2 - 1, innovation++, true)

    return genome;
  }

  const genome = createGenomeWithHidden();

  let network = new Network({ input: 2, output: 1, learningRate: .1 }, genome);

  const XOR = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] }
  ]
  const AND = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [0] },
    { input: [1, 0], output: [0] },
    { input: [1, 1], output: [1] }
  ]
  
  _.each(XOR, example => {
    const result = network.activate(example.input);
    console.log('XOR', example.input, example.output[0], result[0])
  });
  console.log('##############################')


  Trainer.train(network, XOR, 1000);


  console.log('##############################')
  _.each(XOR, example => {
    const result = network.activate(example.input);
    console.log('XOR', example.input, example.output[0], Math.round(result[0] * 10000) / 10000)
  });

  Port.export(network)
});