import Flex from "@/style/Flex"
import { CommentRounded } from "@mui/icons-material"
import { Card } from "../atom/Card"
import Details from "../atom/Details"
import H3 from "../atom/H3"
import IAi from "@/types/data/IAi"
import { useNavigate } from "react-router"
import AutoGPTAPI from "@/api/AutoGPTAPI"

interface ITaskCard {
	$active?: boolean
	ai: IAi
}
const TaskCard = ({ $active, ai }: ITaskCard) => {
	const navigate = useNavigate()
	return (
		<Card
			$textColor={$active ? "primary300" : "grey100"}
			$borderColor={$active ? "primary" : ""}
			$color="grey900"
			elevation={0}
			$active={$active}
			onClick={() => {
				AutoGPTAPI.createInitData({
					ai_role: ai.role,
					ai_name: ai.name,
					ai_goals: ai.goals,
					continuous: ai.continuous,
				})
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
				<Details $color="grey300">{ai.role}</Details>
			</Flex>
		</Card>
	)
}

export default TaskCard
