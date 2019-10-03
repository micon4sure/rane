import Neuron from '../src/Neuron';
import {NEURON_TYPE} from '../src/Neuron';
import squash from '../src/squash';
import * as _ from 'lodash'

test('testNewNeuron', () => {
  return;
  const neuron = new Neuron(1, NEURON_TYPE.hidden, 0.1, 'sigmoid');

  expect(neuron.getId()).toBe(1);
  expect(neuron.getBias()).toBe(.1);
  expect(neuron.getType()).toBe(NEURON_TYPE.hidden);
  expect(typeof neuron.getSquash()).toBe('function');
})