module.exports = {

  preset: 'ts-jest',
  "roots": [
    "<rootDir>/src",
    "<rootDir>/test"
  ],
  "testRegex": "./test/.*.ts$",
  rootDir: '.',
  collectCoverage: true,

  coverageReporters: ["json", "html"]
}