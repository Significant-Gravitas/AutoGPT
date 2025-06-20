import { defineConfig } from "orval";

export default defineConfig({
  library: {
    input: {
      target: `http://localhost:8006/openapi.json`,
      filters: {
        tags: ["library"],
      },

    },
    output: {
      workspace: "./src/api",
      target: `./client/library/library.ts`,
      schemas: "./model",
      client: "react-query",
      httpClient: "fetch",

    },
    hooks: {
      afterAllFilesWrite: 'prettier --write',
    },
  },
});

