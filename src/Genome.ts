import Neuron from './Neuron'
import { NEURON_TYPE } from './Neuron'
import Connection from './Connection';
class Genome {
  public nodes = [];
  public connections = [];

  addNeuron(neuron: Neuron) {
    this.nodes.push(neuron.toJSON())
  }
  addNodeGene(id: number, type: NEURON_TYPE, bias: number, squash: string = 'logistic', enabled: boolean = true) {
    this.nodes.push({
      id,
      type,
      bias,
      squash,
      enabled
    });
  }

  addConnection(connection: Connection) {
    this.connections.push(connection.toJSON());
  }
  addConnectionGene(from: number, to: number, weight: number, innovation: number, enabled: boolean = true) {
    this.connections.push({
      from,
      to,
      weight,
      innovation,
      enabled
    });
  }
}
export default Genome;

export const emptyGenome = (input, output): Genome => {
  const genome = new Genome();
  for (let i = 0; i < input; i++) {
    genome.addNodeGene(i, NEURON_TYPE.input, Math.random() * 2 - 1)
  }
  for (let i = input; i < input + output; i++) {
    genome.addNodeGene(i, NEURON_TYPE.output, Math.random() * 2 - 1)
  }

  let innovation = 0;
  for (let i = 0; i < input; i++) {
    for (let j = input; j < input + output; j++) {
      genome.addConnectionGene(genome.nodes[i].id, genome.nodes[j].id, Math.random() * 2 - 1, innovation++);
    }
  }

  return genome;
};