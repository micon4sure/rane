import Node from './Node'
import { NODE_TYPE } from './Node'

import Connection from './Connection'

import Genome from './Genome';
import { emptyGenome } from './Genome';

import Memory from './Memory'

import * as _ from 'lodash'


class Network {
  private nodeMap = {};
  private inputNodes = Array<Node>();
  private hiddenNodes = Array<Node>();
  private outputNodes = Array<Node>();
  private connections = Array<Connection>();

  private junkGenes = {
    nodes: [],
    connections: []
  }

  private config = {
    learningRate: .001,
    decayRate: .999
  } as any;

  constructor(config = {} as any, genome: Genome = null) {
    _.defaults(config, this.config);
    this.config = config;

    if (genome === null) {
      genome = emptyGenome(config.input, config.output);
    }

    // add nodes
    _.each(genome.nodes, gene => {
      if (!gene.enabled) {
        this.junkGenes.nodes.push(gene);
        return;
      }
      const node = new Node(gene.id, gene.type, gene.bias, gene.squash);
      this.nodeMap[gene.id] = node;
      switch (gene.type) {
        case NODE_TYPE.input:
          this.inputNodes.push(node)
          break;
        case NODE_TYPE.hidden:
          this.hiddenNodes.push(node)
          break;
        case NODE_TYPE.output:
          this.outputNodes.push(node)
          break;
      }
    });

    // add connections
    _.each(genome.connections, gene => {
      const fromNode = this.nodeMap[gene.from];
      const toNode = this.nodeMap[gene.to];
      if (!gene.enabled) {
        this.junkGenes.connections.push(gene);
        return;
      }
      const connection = new Connection(fromNode, toNode, gene.weight, gene.innovation);
      this.connections.push(connection);
      fromNode.connectForward(connection);
      toNode.connectBackward(connection);
    });
  }

  activate(pattern: Array<number>) {
    const memory = new Memory();
    if (pattern.length != this.inputNodes.length) {
      throw new Error('Invalid pattern supplied.')
    }

    _.each(pattern, (activation, i) => {
      this.inputNodes[i].activate(activation, memory);
    })

    return this.getOutput();
  }

  getOutput() {
    const output = [];
    _.each(this.outputNodes, node => {
      output.push(node.getActivation())
    })
    return output;
  }

  train(example) {
    this.activate(example.input);

    _.each(this.outputNodes, (node: Node, index) => {
      node.propagateOutput(example.output[index], this.config.learningRate);
    });
    _.each(this.connections, (connection: Connection, index) => {
      connection.adjust();
    });

    return this.getOutput();
  }

  getConfig() { return this.config; }
  getInputNodes() { return this.inputNodes; }
  getHiddenNodes() { return this.hiddenNodes; }
  getOutputNodes() { return this.outputNodes; }

  getGenome(): Genome {
    const genome = new Genome();

    // add nodes to genome
    _.each(this.nodeMap, (node: Node) => {
      genome.addNode(node)
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