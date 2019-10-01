import Neuron from '../src/Neuron';
import Network from '../src/Network'
import Port from '../src/Port'
import * as _ from 'lodash'


test('testExport/Import', () => {
  const original = new Network({input: 2, output: 1});
  const export_ = original.export();
  const cloned = Network.fromExport(export_);

  const AND = [{ input: [0, 0], output: [0] },
  { input: [0, 1], output: [0] },
  { input: [1, 0], output: [0] },
  { input: [1, 1], output: [1] }]


  Port.export(original);
  
  _.each(AND, example => {
    expect(original.activate(example.input)).toEqual(cloned.activate(example.input));
  }); 

});