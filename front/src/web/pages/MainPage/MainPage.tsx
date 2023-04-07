import { useState } from "react"
import Flex from "../../style/Flex"
import { Button, Comment, Grid, Input } from "./MainPage.styled"

const MainPage = () => {
  const [output, setOutput] = useState([
    {
      title: "Welcome back! ",
      content: "Would you like me to return to being Entrepreneur-GPT?",
    },
    { title: "Using memory of type: PineconeMemory", content: "" },
    { title: "Thinking...", content: "" },
    {
      title: "ENTREPRENEUR-GPT THOUGHTS:",
      content:
        "As an entrepreneur AI, I should first start developing a business plan. I can use my Google search to gather data and inform my approach. ",
    },
    {
      title: "REASONING:",
      content:
        "Designing a thoughtful business plan helps me to ensure that I am executing on a well-founded plan. Conducting some research through Google can inform my initial approach to create an efficient business model. ",
    },
    { title: "PLAN:", content: "" },
    {
      title: "- ",
      content:
        "Use my Google search command to evaluate market trends and determine business strategies.",
    },
    {
      title: "- ",
      content:
        "Develop my initial business plan in writing and store it using write_to_file command.",
    },
    {
      title: "CRITICISM:",
      content:
        "My goal should be airtight. I will have to ensure that any information I obtain over the internet is relevant for my business plan. ",
    },
    {
      title: "NEXT ACTION: ",
      content:
        "COMMAND = google ARGUMENTS = {'input': 'market trends for new businesses'}",
    },
  ])

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
    <Grid>
      <div></div>
      <div>
        <Button
          onClick={() => {
            startScript()
          }}
        >
          start script
        </Button>
        <Button
          onClick={() => {
            fetchData()
          }}
        >
          fetch data
        </Button>
        <Button
          onClick={() => {
            killScript()
          }}
        >
          kill script
        </Button>
        <Flex direction="column" gap={0.5}>
          {output.map((line) => (
            <Comment>{JSON.stringify(line)}</Comment>
          ))}
          <Input placeholder="your input" />
        </Flex>
      </div>
      <div />
    </Grid>
  )
}

export default MainPage
