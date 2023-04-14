import useAgents from "@/hooks/data/useAgents"
import useAiHistory from "@/hooks/data/useAiHistory"
import { Archive, Delete, PlayArrow, Stop } from "@mui/icons-material"
import CommentRoundedIcon from "@mui/icons-material/CommentRounded"
import { Chip } from "@mui/material"
import { useEffect, useRef, useState } from "react"
import { useNavigate, useParams } from "react-router"
import AutoGPTAPI from "../../api/AutoGPTAPI"
import AgentCard from "../../components/molecules/AgentCard"
import SearchInput from "../../components/molecules/SearchInput"
import TaskCard from "../../components/molecules/TaskCard"
import Answers from "../../components/organisms/Answers"
import useAnswerInterceptor from "../../hooks/useAnswerInterceptor"
import Flex from "../../style/Flex"
import SButton from "../../style/SButton"
import IAnswer from "../../types/data/IAnswer"
import {
  ActionBar,
  CommentContainer,
  Container,
  Discussion,
  Grid,
  Input,
  InputContainer,
  LeftContent,
  SIconButton,
} from "./MainPage.styled"
import useAutoGPTAPI from "@/hooks/useAutoGPTAPI"
import SChip from "@/components/atom/SChip"
import AiList from "@/components/organisms/AiList/AiList"
import { useDispatch } from "react-redux"
import { deleteAi } from "@/redux/data/dataReducer"

const MainPage = () => {
  const { aiHistoryArray, aiHistory } = useAiHistory()
  const { id } = useParams<{ id: string }>()
  const { agents } = useAgents()
  const [playing, setPlaying] = useState(false)
  const dispatch = useDispatch()
  const commentsEndRef = useRef(null)
  const navigate = useNavigate()

  const { fetchData } = useAutoGPTAPI()

  useEffect(() => {
    scrollToBottom()
  }, [aiHistoryArray])

  useEffect(() => {
    const interval = setInterval(async () => {
      if (playing) {
        fetchData()
      }
    }, 500)
    return () => clearInterval(interval)
  }, [playing])

  if (!id) {
    navigate("/")
    return null
  }
  const currentAi = aiHistory[id]

  // This function will scroll to the bottom of the messages element
  const scrollToBottom = () => {
    if (!commentsEndRef.current) return
    // @ts-ignore
    commentsEndRef.current.scrollIntoView({ behavior: "smooth" })
  }

  return (
    <Container>
      <Grid>
        <AiList />
        <Discussion>
          <Flex direction="column" gap={0.5}>
            <ActionBar>
              <Flex justify="space-between" align="center" fullWidth>
                <Flex gap={0.5} align="center">
                  <CommentRoundedIcon />
                  <h2>{currentAi.name}</h2>
                  <Chip label="Continuous" color="primary" size="small" />
                </Flex>
                <Flex gap={0.5} align="center">
                  <SIconButton
                    onClick={() => {
                      if (playing) {
                        setPlaying(!playing)
                        AutoGPTAPI.killScript()
                      }
                      navigate("/")
                      dispatch(deleteAi(id))
                    }}
                  >
                    <Delete fontSize="small" />
                  </SIconButton>
                  <SIconButton>
                    <Archive fontSize="small" />
                  </SIconButton>
                </Flex>
              </Flex>
              <div>Role : {currentAi.role}</div>
            </ActionBar>
            <CommentContainer>
              <Answers answers={currentAi.answers} playing={playing} />
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
        <LeftContent>
          <Flex direction="column" gap={0.5}>
            <h2>Your goals</h2>
            {(currentAi.goals ?? []).map((goal) => {
              if (goal === "") return null
              return (
                <SChip
                  $color="primary300"
                  key={goal}
                  label={goal}
                  color="primary"
                  variant="outlined"
                  sx={{ mr: 0.5 }}
                />
              )
            })}
          </Flex>
          <Flex direction="column" gap={0.5}>
            <h2>Your agents</h2>
            {currentAi.agents.map((agentName) => {
              const agent = agents[agentName]
              return <AgentCard key={agent.name} agent={agent} />
            })}
          </Flex>
        </LeftContent>
      </Grid>
    </Container>
  )
}

export default MainPage
