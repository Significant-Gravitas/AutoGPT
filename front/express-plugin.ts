export default function express(path: string) {
  return {
    name: "vite3-plugin-express",
    configureServer: async (server: any) => {
      server.middlewares.use(async (req: any, res: any, next: any) => {
        process.env["VITE"] = "true"
        try {
          const { app } = await server.ssrLoadModule(path)
          app(req, res, next)
        } catch (err) {
          console.error(err)
        }
      })
    },
  }
}
