import Network from './Network'

class NEAT {

  private input:number;
  private output:number;
  private populationCount: number;
  private networkConfig: any;

  private population = new Array<Network>();

  constructor(input: number, output: number, populationCount: number, networkConfig: any) {
    for(let i = 0; i < populationCount; i++) {
      this.population.push(new Network(networkConfig));
    }
  }
}

export default NEAT;