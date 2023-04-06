import { useState } from "react"
import reactLogo from "./assets/react.svg"
import viteLogo from "/vite.svg"
import { useEffect } from "react"
import "./App.css"
//Add this function
const getGreeting = async function () {
  const res = await fetch("/api/python")
  return await res.json()
}

function App() {
  const [greeting, setGreeting] = useState("")

  const callScript = () => {
    getGreeting().then((res) => setGreeting(res.greeting))
  }

  return (
    <>
      <button
        onClick={() => {
          callScript()
        }}
      >
        call script
      </button>
      <p>Server respons2azee: {greeting}</p>
    </>
  )
}

export default App
