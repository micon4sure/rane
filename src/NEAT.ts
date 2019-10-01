import Network from './Network'

class NEAT {

  private populationCount: number;
  private networkConfig: any;

  private population = new Array<Network>();

  constructor(populationCount: number, networkConfig: any) {
    this.populationCount = populationCount;
    this.networkConfig = networkConfig;
    for(let i = 0; i < populationCount; i++) {
      this.population.push(new Network(networkConfig));
    }
  }
}

export default NEAT;