import { useState } from "react"
import reactLogo from "./assets/react.svg"
import viteLogo from "/vite.svg"
import { useEffect } from "react"
import Flex from "./style/Flex"

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
    // remove last char from data data is a string
    // remove \n
    data = data.output.replace("\n", "")
    data = data.slice(0, -2)
    console.log(data)
    const json = JSON.parse(`[${data}]`)
    setOutput(json)
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
      <Flex direction="column" gap={0.5}>
        {output.map((line) => (
          <div>{JSON.stringify(line)}</div>
        ))}
      </Flex>
    </>
  )
}

export default App
