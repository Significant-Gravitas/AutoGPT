import { ArrowForwardIos, SmartToy } from "@mui/icons-material"
import Flex from "../../../style/Flex"
import { Card } from "../atom/Card"
import H3 from "../atom/H3"
import Details from "../atom/Details"
import IAgent from "../../../types/data/IAgent"
import styled from "styled-components"

// console like looking
const Prompt = styled.div`
  background-color: var(--grey800);
  color: var(--grey100);
  padding: 0.5rem;
  border-radius: 0.5rem;
  font-family: "Roboto Mono", monospace;
  font-size: 0.8rem;
`

const AgentCard = ({ agent }: { agent: IAgent }) => {
  return (
    <Card elevation={0}>
      <Flex direction="column" gap={1}>
        <Flex justify="space-between" align="center">
          <Flex gap={0.5} align="center">
            <SmartToy fontSize="small" />
            <H3>{agent.name}</H3>
          </Flex>
          <ArrowForwardIos />
        </Flex>
        <Details>{agent.task}</Details>
        <Prompt>{agent.prompt}</Prompt>
      </Flex>
    </Card>
  )
}

export default AgentCard
