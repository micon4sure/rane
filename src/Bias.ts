class Bias {

  public id;
  public value;
  public adjustment = 0;
  public delta = 0;

  constructor(id, initial = 1) {
    this.id = id;
    this.value = initial;
  }
}

export default Bias;