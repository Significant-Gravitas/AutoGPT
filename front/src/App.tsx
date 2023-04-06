import { useState } from "react"
import reactLogo from "./assets/react.svg"
import viteLogo from "/vite.svg"
import { useEffect } from "react"
import "./App.css"

function App() {
  const [output, setOutput] = useState([])

  const startScript = async () => {
    const res = await fetch("/api/start")
    return await res.json()
  }
  const killScript = async () => {
    const res = await fetch("/api/stop")
    return await res.json()
  }
  const fetchData = async () => {
    const res = await fetch("/api/data")
    let data = await res.json()
    const lines = data.output.split("\n")
    setOutput(lines)
  }

  return (
    <>
      <button
        onClick={() => {
          startScript()
        }}
      >
        start script
      </button>
      <button
        onClick={() => {
          fetchData()
        }}
      >
        fetch data
      </button>
      <button
        onClick={() => {
          killScript()
        }}
      >
        kill script
      </button>
      <ul>
        {output.map((line) => (
          <li>{line}</li>
        ))}
      </ul>
    </>
  )
}

export default App
