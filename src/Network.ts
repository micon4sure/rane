import Neuron from './Neuron'
import { NEURON_TYPE } from './Neuron'

import Connection from './Connection'

import Genome from './Genome';
import { emptyGenome } from './Genome';

import Memory from './Memory'

import Trainer from './Trainer'

import * as _ from 'lodash'


class Network {
  private neuronMap = {};
  private inputNeurons = Array<Neuron>();
  private hiddenNeurons = Array<Neuron>();
  private outputNeurons = Array<Neuron>();
  private connections = Array<Connection>();

  private junkGenes = {
    nodes: [],
    connections: []
  }

  private config = {
    learningRate: .001,
    decayRate: .999
  } as any;

  private activations = 0;

  constructor(config = {} as any, genome: Genome = null) {
    _.defaults(config, this.config);
    this.config = config;

    if (genome === null) {
      genome = emptyGenome(config.input, config.output);
    }

    // add neurons
    _.each(genome.nodes, gene => {
      if (!gene.enabled) {
        this.junkGenes.nodes.push(gene);
        return;
      }
      const neuron = new Neuron(gene.id, gene.type, gene.bias);
      this.neuronMap[gene.id] = neuron;
      switch (gene.type) {
        case NEURON_TYPE.input:
          this.inputNeurons.push(neuron)
          break;
        case NEURON_TYPE.hidden:
          this.hiddenNeurons.push(neuron)
          break;
        case NEURON_TYPE.output:
          this.outputNeurons.push(neuron)
          break;
      }
    });

    // add connections
    _.each(genome.connections, gene => {
      const fromNeuron = this.neuronMap[gene.from];
      const toNeuron = this.neuronMap[gene.to];
      if (!gene.enabled) {
        this.junkGenes.connections.push(gene);
        return;
      }
      const connection = new Connection(fromNeuron, toNeuron, gene.weight, gene.innovation);
      this.connections.push(connection);
      fromNeuron.connectForward(connection);
      toNeuron.connectBackward(connection);
    });
  }

  activate(pattern: Array<number>) {
    const memory = new Memory();
    if (pattern.length != this.inputNeurons.length) {
      throw new Error('Invalid pattern supplied.')
    }

    _.each(pattern, (activation, i) => {
      this.inputNeurons[i].activate(activation, memory);
    })

    const result = [];
    _.each(this.outputNeurons, neuron => {
      result.push(neuron.getActivation())
    })
    return result;
  }

  train(data, iterations) {
    Trainer.train(this, data, iterations);
  }

  getConfig() { return this.config; }
  getInputNeurons() { return this.inputNeurons; }
  getHiddenNeurons() { return this.hiddenNeurons; }
  getOutputNeurons() { return this.outputNeurons; }

  getGenome(): Genome {
    const genome = new Genome();

    // add neurons to genome
    _.each(this.neuronMap, (neuron: Neuron) => {
      genome.addNeuron(neuron)
    });
    _.each(this.junkGenes.nodes, gene => {
      genome.addNodeGene(gene.id, gene.type, gene.bias, gene.squash, false);
    });

    // add connections to genome
    _.each(this.connections, (connection: Connection) => {
      genome.addConnection(connection);
    });
    _.each(this.junkGenes.connections, gene => {
      genome.addConnectionGene(gene.from, gene.to, gene.weight, gene.innovation, false);
    });
    return genome;
  }
  export() {
    return {
      config: this.config,
      genome: this.getGenome()
    }
  }
  static fromExport(export_) {
    return new Network(export_.config, export_.genome);
  }
}

export default Network;