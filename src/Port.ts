import Network from './Network'
import Neuron from './Neuron'
import * as fs from 'fs';
import * as _ from 'lodash'

class Port {
  static export(network) {
    fs.writeFile('../rane-vis/data.json', JSON.stringify(network.export()), (err) => {
      if (err) throw err;
    });
  }
}

export default Port;