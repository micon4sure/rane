class Log {

  public static type = {
    DERIVATIVES: 1,
    OUTPUT: 2,
    HIDDEN: 4,
    WEIGHTS: 8,
    BIAS: 16
  };
  private static types;
  private static mask = Log.type.DERIVATIVES |  Log.type.OUTPUT | Log.type.HIDDEN | Log.type.WEIGHTS |Log.type.BIAS;
  
  public static ALL = Log.mask;
  public static NONE = 0;

  public static setTypes(types) {
    Log.types = types;
  }

  public static log(type, ...payload) {
    if(type = Log.types & type) {
      console.log(...payload)
    }
  }
}

export default Log;