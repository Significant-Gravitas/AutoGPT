import { ArrowForwardIos, SmartToy } from "@mui/icons-material"
import Flex from "../../../style/Flex"
import { Card } from "../atom/Card"
import H3 from "../atom/H3"
import Details from "../atom/Details"

const AgentCard = () => {
  return (
    <Card elevation={0}>
      <Flex direction="column" gap={1}>
        <Flex justify="space-between" align="center">
          <Flex gap={0.5} align="center">
            <SmartToy fontSize="small" />
            <H3>Task 2</H3>
          </Flex>
          <ArrowForwardIos />
        </Flex>
        <Details>
          Use my Google search command to evaluate market trends and determine
          business strategies.
        </Details>
      </Flex>
    </Card>
  )
}

export default AgentCard
