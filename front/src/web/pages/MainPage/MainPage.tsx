import { useState } from "react"
import Flex from "../../style/Flex"
import {
  ActionBar,
  CardTask,
  Comment,
  CommentContainer,
  Container,
  Discussion,
  Grid,
  Input,
  InputContainer,
  Line,
  RightTasks,
  Search,
} from "./MainPage.styled"
import SButton from "../../style/SButton"
import { TextField } from "@mui/material"

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
    <Container>
      <Grid>
        <RightTasks>
          <Flex direction="column" gap={0.5}>
            <h2>Your AI</h2>
            <Search>
              <TextField
                id="outlined-basic"
                label="Outlined"
                variant="outlined"
              />
            </Search>
            <CardTask elevation={0}>
              <Flex justify="space-between" align="center">
                <h3>Task 1</h3>
                <div>7 Apr</div>
              </Flex>
              <p>
                Use my Google search command to evaluate market trends and
                determine business strategies.
              </p>
            </CardTask>
            <CardTask elevation={0}>
              <Flex justify="space-between" align="center">
                <h3>Task 2</h3>
                <div>6 Apr</div>
              </Flex>
              <p>
                Use my Google search command to evaluate market trends and
                determine business strategies.
              </p>
            </CardTask>
            <SButton $color="yellow300" variant="outlined">
              Create a new Ai
            </SButton>
          </Flex>
        </RightTasks>
        <Discussion>
          <Flex direction="column" gap={0.5}>
            <ActionBar>
              <SButton
                $color="yellow"
                variant="contained"
                onClick={() => {
                  startScript()
                }}
              >
                start script
              </SButton>
              <SButton
                $color="yellow"
                onClick={() => {
                  fetchData()
                }}
              >
                fetch data
              </SButton>
              <SButton
                $color="yellow"
                onClick={() => {
                  killScript()
                }}
              >
                kill script
              </SButton>
            </ActionBar>
            <CommentContainer>
              {output.map((line) => (
                <>
                  <Line content={line.title} />
                  <Comment>
                    {line.content === "" ? line.title : line.content}
                  </Comment>
                </>
              ))}
            </CommentContainer>
            <InputContainer>
              <Input placeholder="your input" />
            </InputContainer>
          </Flex>
        </Discussion>
        <RightTasks>
          <Flex direction="column" gap={0.5}>
            <h2>Your agents</h2>
            <CardTask>
              <Flex justify="space-between" align="center">
                <h3>Agent - 1</h3>
                <div>6 Apr</div>
              </Flex>
              <p>Scrapping</p>
            </CardTask>
            <CardTask>
              <Flex justify="space-between" align="center">
                <h3>Agent - 2</h3>
                <div>6 Apr</div>
              </Flex>
              <p>Searching</p>
            </CardTask>
            <CardTask>
              <Flex justify="space-between" align="center">
                <h3>Agent - 3</h3>
                <div>6 Apr</div>
              </Flex>
              <p>Writing</p>
            </CardTask>
          </Flex>
        </RightTasks>
      </Grid>
    </Container>
  )
}

export default MainPage
