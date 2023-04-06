import express from "express"
import { spawn } from "child_process"

export const app = express()

if (!process.env["VITE"]) {
  const frontendFiles = process.cwd() + "/dist"
  app.use(express.static(frontendFiles))

  app.get("/*", (_, res) => {
    res.send(frontendFiles + "/index.html")
  })
  app.listen(process.env["PORT"])
}
app.get("/api/test", (_, res) => res.json({ greeting: "Hellazeo" }))
app.get("/api/python", (req, res) => {
  let dataToSend = ""

  console.log("Python script called")
  const python = spawn("python", ["script1.py"])
  // collect data from script
  python.stdout.on("data", function (data) {
    console.log("Pipe data from python script ...")
    dataToSend = data.toString()
  })
  // in close event we are sure that stream from child process is closed
  python.on("close", (code) => {
    console.log(`child process close all stdio with code ${code}`)
    // send data to browser
    res.json({ greeting: dataToSend })
  })
})
app.listen(3001, () => console.log("Server running on port 3000"))
