// `?raw` imports a file's text, e.g. the token stories reading the @theme
// blocks of globals.css. Storybook and Vitest both resolve the query.
declare module "*?raw" {
  const content: string;
  export default content;
}
