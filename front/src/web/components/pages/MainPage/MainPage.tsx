import { Archive, Delete, Pause, PlayArrow, Stop } from "@mui/icons-material"
import CommentRoundedIcon from "@mui/icons-material/CommentRounded"
import { Chip } from "@mui/material"
import { useEffect, useRef, useState } from "react"
import Flex from "../../../style/Flex"
import SButton from "../../../style/SButton"
import AgentCard from "../../UI/molecules/AgentCard"
import SearchInput from "../../UI/molecules/SearchInput"
import TaskCard from "../../UI/molecules/TaskCard"
import Comments from "../../UI/organisms/Comments"
import {
  ActionBar,
  CommentContainer,
  Container,
  Discussion,
  Grid,
  Input,
  InputContainer,
  RightTasks,
  SIconButton,
} from "./MainPage.styled"
import AutoGPTAPI from "../../../api/AutoGPTAPI"

const MainPage = () => {
  const [playing, setPlaying] = useState(false)
  const commentsEndRef = useRef(null)

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

  // This function will scroll to the bottom of the messages element
  const scrollToBottom = () => {
    commentsEndRef.current.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [output])

  // AutoGpt.fetchData every 5 seconds if playing
  useEffect(() => {
    const interval = setInterval(async () => {
      if (playing) {
        const data = await AutoGPTAPI.fetchData()
        setOutput(data)
      }
    }, 5000)
    return () => clearInterval(interval)
  }, [playing])

  return (
    <Container>
      <Grid>
        <RightTasks>
          <Flex direction="column" gap={1}>
            <h2>All your AI</h2>
            <SearchInput />
            <TaskCard />
            <TaskCard $active />
            <TaskCard />
            <TaskCard />
            <SButton $color="yellow300" variant="outlined">
              Create a new Ai
            </SButton>
          </Flex>
        </RightTasks>
        <Discussion>
          <Flex direction="column" gap={0.5}>
            <ActionBar>
              <Flex justify="space-between" align="center" fullWidth>
                <Flex gap={0.5} align="center">
                  <CommentRoundedIcon />
                  <h2>Ai Name</h2>
                  <Chip label="Continuous" color="primary" size="small" />
                </Flex>
                <Flex gap={0.5} align="center">
                  <SIconButton>
                    <Delete fontSize="small" />
                  </SIconButton>
                  <SIconButton>
                    <Archive fontSize="small" />
                  </SIconButton>
                </Flex>
              </Flex>
              <div>
                Ai role is lorem ipsum dolor sit amet, consectetur adipiscing
                elit. Aliquam at ipsum eu nunc commodo posuere et sit amet
                ligula. Aenean quis rhoncus nunc, quis interdum justo. Duis quis
                nisl
              </div>
            </ActionBar>
            <CommentContainer>
              <Comments comments={output} />
              <div ref={commentsEndRef} />
            </CommentContainer>
            <InputContainer>
              <Input placeholder="your input" />
              {!playing && (
                <SIconButton
                  onClick={() => {
                    setPlaying(!playing)
                    AutoGPTAPI.startScript()
                  }}
                >
                  <PlayArrow />
                </SIconButton>
              )}
              {playing && (
                <SIconButton
                  onClick={() => {
                    setPlaying(!playing)
                    AutoGPTAPI.killScript()
                  }}
                >
                  <Stop />
                </SIconButton>
              )}
            </InputContainer>
          </Flex>
        </Discussion>
        <RightTasks>
          <Flex direction="column" gap={0.5}>
            <h2>Your agents</h2>
            <AgentCard />
            <AgentCard />
            <AgentCard />
          </Flex>
        </RightTasks>
      </Grid>
    </Container>
  )
}

export default MainPage
