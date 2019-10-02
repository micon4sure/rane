import Network from './Network'
import Trainer from './Trainer'
import * as _ from 'lodash'
import Genome from './Genome';
import {NEURON_TYPE} from './Neuron'
import Squash from './squash';

class NEAT {

  private populationSize: number;
  private elitism: number;
  private networkConfig: any;
  private innovation: number;

  private population = new Array<Network>();

  constructor(populationCount: number, elitism: number, networkConfig: any) {
    this.populationSize = populationCount;
    this.elitism = elitism;
    this.networkConfig = networkConfig;
    for (let i = 0; i < populationCount; i++) {
      this.population.push(new Network(networkConfig));
    }

    this.innovation = networkConfig.input * networkConfig.output;
  }

  train(data, iterations, generations): Network {
    let alpha: Network;
    for (let i = 0; i < generations; i++) {
      console.log('GENERATION', i)
      const errors = [];
      _.each(this.population, (network: Network) => {
        const error = Trainer.train(network, data, iterations);
        errors.push({ error, network })
      })

      // sort networks by least error
      const sorted = _.sortBy(errors, result => {
        return Math.abs(result.error);
      });

      const newPopulation = [];
      alpha = sorted[0].network;

      // add the elite unchanged
      for (let j = 0; j < this.elitism; j++) {
        newPopulation.push(sorted[j].network);
      }

      // produce offspring off the elite
      for (let j = 0; j < this.elitism; j += 2) {
        const male = newPopulation[j];
        const female = newPopulation[j + 1];

        const offspring = this.breed(male, female);
        newPopulation.push(offspring);
      }

      while (newPopulation.length < this.populationSize) {
        newPopulation.push(this.mutate(newPopulation[0]));
      }

      this.population = newPopulation;
    }
    return alpha;
  }

  breed(male: Network, female: Network) {
    const maleGenome = male.export().genome as Genome;
    const femaleGenome = female.export().genome as Genome;

    const nodes = {};
    const connections = {};

    _.each(maleGenome.nodes, node => {
      nodes[node.id] = node
    })
    _.each(femaleGenome.nodes, node => {
      nodes[node.id] = node
    })

    _.each(maleGenome.connections, connection => {
      connections[connection.id] = connection
    })
    _.each(femaleGenome.connections, connection => {
      connections[connection.id] = connection
    })


    const offspringGenome = new Genome();
    _.each(nodes, (node: any) => {
      offspringGenome.addNodeGene(node.id, node.type, node.bias, node.squash, node.enabled);
    })
    _.each(connections, (connection: any) => {
      offspringGenome.addConnectionGene(connection.from, connection.to, connection.weight, connection.innovation, connection.enabled);
    })

    return new Network(this.networkConfig, offspringGenome);
  }

  mutate(network: Network) {
    const genome = network.getGenome();
    let mutated = genome;

    if (this.getRandomBoolean()) {
      mutated = this.addNode(genome);
    }
    if (this.getRandomBoolean()) {
      mutated = this.addConnection(genome);
    }

    return new Network(network.getConfig(), mutated);
  }

  addNode(genome: Genome) {
    const index = this.getRandomInt(0, genome.connections.length - 1);
    const connection = genome.connections[index];
    const nodeId = genome.nodes.length;

    const squash = Squash[this.getRandomInt(0, Object.keys(Squash).length - 1)]
    genome.addNodeGene(nodeId, NEURON_TYPE.hidden, Math.random() * 2 - 1, squash)

    genome.addConnectionGene(connection.from, nodeId, Math.random(), ++this.innovation, true);
    genome.addConnectionGene(nodeId, connection.to, 1, ++this.innovation, true);

    connection.enabled = false;
    return genome;
  }

  addConnection(genome: Genome) {
    let from = null;
    let to = null;
    while(from === null || to === null) {
      const fromCandidate = genome.nodes[this.getRandomInt(0, genome.nodes.length - 1)]
      const toCandidate = genome.nodes[this.getRandomInt(0, genome.nodes.length - 1)]

      if(fromCandidate.type != NEURON_TYPE.output) {
        from = fromCandidate;
      }
      if(toCandidate.type != NEURON_TYPE.input) {
        to = toCandidate;
      }

      // avoid backwards connection

      if(from !== null && to !== null) {
        // avoid self connections
        //TODO: enable self connections but handle them inside neuron
        if(from.id == to.id) {
          from = to = null;
        }
      } 
    }
    genome.addConnectionGene(from.id, to.id, Math.random(), ++this.innovation, true);
    return genome;
  }

  getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }
  getRandomBoolean() {
    return Boolean(Math.round(Math.random()));
  }
}

export default NEAT;