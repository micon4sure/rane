module.exports = {

  "roots": [
    "<rootDir>/src",
    "<rootDir>/test"
  ],
  "testRegex": "./test/.*.ts$",
  "rootDir": ".",
  transform: {
    "^.+\\.tsx?$": "ts-jest",
  },
  collectCoverage: true,

  coverageReporters: ["json", "html"]
}