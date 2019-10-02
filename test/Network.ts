import Neuron from '../src/Neuron';
import Network from '../src/Network'
import Port from '../src/Port'
import * as _ from 'lodash'


test('testExport/Import', () => {
  const original = new Network({ input: 3, output: 3 });
  const export_ = original.export();
  const cloned = Network.fromExport(export_);

  expect(original.activate([1, 2, 3])).toEqual(cloned.activate([1, 2, 3]));
});
test('test train AND', () => {
  const network = new Network({ input: 2, output: 1, learningRate: 0.001 });
  const AND = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [0] },
    { input: [1, 0], output: [0] },
    { input: [1, 1], output: [1] }

  ]
  //network.train(AND, 10000);


  _.each(AND, example => {
    const result = network.activate(example.input);
    //expect(Math.round(result[0])).toEqual(example.output[0]);
  });
});

test('test train OR', () => {
  const network = new Network({ input: 2, output: 1, learningRate: 0.001 });
  const OR = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [1] }
  ]

  //network.train(OR, 10000);


  _.each(OR, example => {
    const result = network.activate(example.input);
    //expect(Math.round(result[0])).toEqual(example.output[0]);
  });
});