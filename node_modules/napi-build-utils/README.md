# napi-build-utils

[![npm](https://img.shields.io/npm/v/napi-build-utils.svg)](https://www.npmjs.com/package/napi-build-utils)
![Node version](https://img.shields.io/node/v/prebuild.svg)
[![Build Status](https://travis-ci.org/inspiredware/napi-build-utils.svg?branch=master)](https://travis-ci.org/inspiredware/napi-build-utils) 
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg)](http://standardjs.com/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

A set of utilities to assist developers of tools that build [N-API](https://nodejs.org/api/n-api.html#n_api_n_api) native add-ons.

## Background

This module is targeted to developers creating tools that build N-API native add-ons. 

It implements a set of functions that aid in determining the N-API version supported by the currently running Node instance and the set of N-API versions against which the N-API native add-on is designed to be built. Other functions determine whether a particular N-API version can be built and can issue console warnings for unsupported N-API versions. 

Unlike the modules this code is designed to facilitate building, this module is written entirely in JavaScript. 

## Quick start

```bash
$ npm install napi-build-utils
```

The module exports a set of functions documented [here](./index.md). For example:

```javascript
var napiBuildUtils = require('napi-build-utils');
var napiVersion = napiBuildUtils.getNapiVersion(); // N-API version supported by Node, or undefined.
```

## Declaring supported N-API versions

Native modules that are designed to work with [N-API](https://nodejs.org/api/n-api.html#n_api_n_api) must explicitly declare the N-API version(s) against which they are coded to build. This is accomplished by including a `binary.napi_versions` property in the module's `package.json` file. For example:

```json
"binary": {
  "napi_versions": [2,3]
}
``` 

In the absence of a need to compile against a specific N-API version, the value `3` is a good choice as this is the N-API version that was supported when N-API left experimental status. 

Modules that are built against a specific N-API version will continue to operate indefinitely, even as later versions of N-API are introduced. 

## Support

If you run into problems or limitations, please file an issue and we'll take a look. Pull requests are also welcome.  
