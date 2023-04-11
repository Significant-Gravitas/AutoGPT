import Flex from "@/style/Flex"
import { CommentRounded } from "@mui/icons-material"
import { Card } from "../atom/Card"
import Details from "../atom/Details"
import H3 from "../atom/H3"

interface ITaskCard {
  $active?: boolean
}
const TaskCard = ({ $active }: ITaskCard) => {
  return (
    <Card elevation={0} $active={$active}>
      <Flex direction="column" gap={1}>
        <Flex justify="space-between" align="center">
          <Flex gap={0.5} align="center">
            <CommentRounded fontSize="small" />
            <H3>Task 1</H3>
          </Flex>
          <div>6 Apr</div>
        </Flex>
        <Details>
          Use my Google search command to evaluate market trends and determine
          business strategies.
        </Details>
      </Flex>
    </Card>
  )
}

export default TaskCard
