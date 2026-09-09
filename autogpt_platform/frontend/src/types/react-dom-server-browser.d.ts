// Next.js forbids importing "react-dom/server" from app-router modules; the
// browser build is the sanctioned way to render static markup from a route
// handler, but @types/react-dom only declares "./server".
declare module "react-dom/server.browser" {
  export * from "react-dom/server";
}
