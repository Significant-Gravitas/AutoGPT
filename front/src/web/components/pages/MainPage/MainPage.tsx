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
import IAnswer from "../../../types/data/IAnswer"
import useAnswerInterceptor from "../../../hooks/useAnswerInterceptor"

const MainPage = () => {
  const [playing, setPlaying] = useState(false)
  const commentsEndRef = useRef(null)
  const [output, setOutput] = useState<IAnswer[]>([])
  const { agents } = useAnswerInterceptor(output)

  // This function will scroll to the bottom of the messages element
  const scrollToBottom = () => {
    if (!commentsEndRef.current) return
    // @ts-ignore
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
        if (data.length === 0) return
        setOutput([...output, ...data])
      }
    }, 500)
    return () => clearInterval(interval)
  }, [playing, output])

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
            {agents.map((agent) => (
              <AgentCard key={agent.name} agent={agent} />
            ))}
          </Flex>
        </RightTasks>
      </Grid>
    </Container>
  )
}

export default MainPage
