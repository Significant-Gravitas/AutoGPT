import express from "express"
import { spawn } from "child_process"
import fs from "fs"
import bodyParser from "body-parser"
import yaml from "js-yaml"

export const app = express()
app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser.json())

if (!process.env["VITE"]) {
  const frontendFiles = `${process.cwd()}/dist`
  app.use(express.static(frontendFiles))
}
// spawn but no start
let python = null as any
let dataToSend = ""

app.get("/api/start", (_, res) => {
  python = spawn("sh", ["./run-web.sh"])
  python.stdout.on("data", function (data: string) {
    console.log(data.toString())
    dataToSend = dataToSend + data.toString()
  })

  python.on("close", (code: string) => {
    console.log(`child process close all stdio with code ${code}`)
  })
  // error
  python.stderr.on("data", (data: string) => {
    console.log(`stderr: ${data}`)
  })

  console.log("Python script started")

  res.json({ output: "Python script started" })
})


app.post("/api/download", (req, res) => {
  const file = `../auto_gpt_workspace/${req.body.filename}`;
  res.download(file); // Set disposition and send it.
});


app.get("/api/data", (req, res) => {
  res.json({ output: dataToSend })
  dataToSend = ""
})

app.get("/api/stop", (_, res) => {
  python.kill()
  res.json({ output: "Python script stopped" })
})

app.post("/api/init", (req, res) => {
  const yamlString = yaml.dump(req.body)
  fs.writeFileSync("../ai_settings.yaml", `${yamlString}`, "utf8")
})

// kill python process on exit
process.on("exit", () => {
  python.kill()
  console.log("Python script killed")
})
app.listen(3001, () => console.log("Server running on port 3001"))

if (!process.env["VITE"]) {
  const frontendFiles = `${process.cwd()}/dist`
  app.get("/*", (_, res) => {
    res.send(`${frontendFiles}/index.html`)
  })

  app.listen(process.env["PORT"])
}
