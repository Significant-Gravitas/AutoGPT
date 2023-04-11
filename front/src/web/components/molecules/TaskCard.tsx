import Flex from "@/style/Flex"
import { CommentRounded } from "@mui/icons-material"
import { Card } from "../atom/Card"
import Details from "../atom/Details"
import H3 from "../atom/H3"
import IAi from "@/types/data/IAi"
import { useNavigate } from "react-router"

interface ITaskCard {
  $active?: boolean
  ai: IAi
}
const TaskCard = ({ $active, ai }: ITaskCard) => {
  const navigate = useNavigate()
  return (
    <Card
      elevation={0}
      $active={$active}
      onClick={() => {
        navigate(`/main/${ai.id}`)
      }}
    >
      <Flex direction="column" gap={1}>
        <Flex justify="space-between" align="center">
          <Flex gap={0.5} align="center">
            <CommentRounded fontSize="small" />
            <H3>{ai.name}</H3>
          </Flex>
          <div>
            {new Date(ai.createdAt).toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
            })}
          </div>
        </Flex>
        <Details>{ai.role}</Details>
      </Flex>
    </Card>
  )
}

export default TaskCard
