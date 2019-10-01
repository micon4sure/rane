// workaround decorator
// TypeScript does not provide .name for functions
function named(target: any, key: string) {
  target[key].functionName = key;
}

export default abstract class Squash {
  @named
  static logistic(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => Math.exp(-x) / (Math.pow((1 + Math.exp(-x)), 2)) : (x) => 1 / (1 + Math.exp(-x));

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static tanh(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => 1 - (Math.tanh(x) * Math.tanh(x)) : (x) => Math.tanh(x)

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static identity(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => 1 : (x) => x;

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static step(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => 0 : (x) => x > 0 ? 1 : 0;

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static relu(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => x > 0 ? 1 : 0 : (x) => x > 0 ? x : 0;

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static softsign(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => x / ((1 + Math.abs(x)) * (1 + Math.abs(x))) : (x) => x / 1 + Math.abs(x);

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static sinusoid(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => Math.cos(x) : (x) => Math.sin(x);

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static gaussian(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => -2 * x * Math.exp(-(x * x)) : (x) => Math.exp(-(x * x));

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static bentIdentity(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => x / (2 * Math.sqrt((x * x) + 1)) + 1 : (x) => (Math.sqrt((x * x) + 1) - 1) / 2 + x;

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static bipolar(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => 0 : (x) => x > 0 ? 1 : -1;

    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static bipoliarSigmoid(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => (2 * Math.exp(-x)) / (Math.pow((1 + Math.exp(-x)), 2)) : (x) => 2 / (1 + Math.exp(-x)) - 1;

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static hard_tanh(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => x > -1 && x < 1 ? 1 : 0 : (x) => Math.max(-1, Math.min(1, x));

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static absolute(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => x < 0 ? -1 : 1 : (x) => Math.abs(x);

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static inverse(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const f = derivate ? (x) => -1 : (x) => 1 - x;

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }

  @named
  static selu(x, derivate) {

    // Dirty but neccessary to support Network.standalone as currently written
    const clamp = function (x) {
      const max = Number.MAX_VALUE

      return x === Infinity
        ? max
        : x === -Infinity
          ? -max
          : x
    }

    if (x == undefined) throw new ReferenceError("Parameter 'x' is required, but it was not defined");

    const alpha = 1.6732632423543772848170429916717;
    const scale = 1.0507009873554804934193349852946;

    const f = derivate ? (x) => x > 0 ? scale : ((x > 0 ? x : alpha * Math.exp(x) - alpha) + alpha) * scale : (x) => (x > 0 ? x : alpha * Math.exp(x) - alpha) * scale;

    // return Array.isArray(x) ? x.map(f) : f(x); unsafe mode
    return Array.isArray(x) ? x.map(clamp(f)) : clamp(f(x));
  }
}