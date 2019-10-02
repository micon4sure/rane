class Memory {
  private activatedConnections = new Array<number>();
  activated(id: number) {
    this.activatedConnections.push(id)
  }

  allowed(id: number) {
    return !this.activatedConnections.includes(id)
  }
}

export default Memory;