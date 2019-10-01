import { NEURON_TYPE } from './Neuron'
class Genome {
  public neurons = [];
  public connections = [];

  addNodeGene(id: string, type: NEURON_TYPE, bias: number, squash: string = 'logistic', enabled: boolean = true) {
    this.neurons.push({
      id,
      type,
      bias,
      squash,
      enabled
    });
  }

  addConnectionGene(from: string, to: string, weight: number, innovation: number, enabled: boolean = true) {
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
    genome.addNodeGene('i' + i, NEURON_TYPE.input, Math.random() * 2 - 1)
  }
  for (let i = 0; i < output; i++) {
    genome.addNodeGene('o' + i, NEURON_TYPE.output, Math.random() * 2 - 1)
  }

  let innovation = 0;
  for (let i = 0; i < input; i++) {
    for (let j = input; j < input + output; j++) {
      genome.addConnectionGene(genome.neurons[i].id, genome.neurons[j].id, Math.random(), innovation++);
    }
  }

  return genome;
};