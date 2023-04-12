import React, { useState } from "react"
import styled from "styled-components"
import DoorBackOutlinedIcon from "@mui/icons-material/DoorBackOutlined"
import Flex from "@/style/Flex"
import { SIconButton } from "@/pages/MainPage/MainPage.styled"

const LeftPanelContainer = styled.div`
  width: 5rem;
  background-color: black;
  color: var(--grey100);
  height: 100vh;
  padding: 1rem;
`
const Avatar = styled.img`
  width: 3rem;
  height: 3rem;
  border-radius: 50%;
  margin-top: 1rem;
`

const LeftPanel = () => {
	const [currentColor, setCurrentColor] = useState("yellow")
	const availableColors = [
		"yellow",
		"grey",
		"purple",
		"emerald",
		"caramel",
		"blue",
		"cyan",
	]

	const changeColor = (color: string) => {
		setCurrentColor(color)
		document.documentElement.style.setProperty(
			"--primary",
			"var(--" + color + ")",
		)
		document.documentElement.style.setProperty(
			"--primary-hue",
			"var(--" + color + "-hue)",
		)
		document.documentElement.style.setProperty(
			"--primary-saturation",
			"var(--" + color + "-saturation)",
		)
		document.documentElement.style.setProperty(
			"--primary-lightness",
			"var(--" + color + "-lightness)",
		)
		document.documentElement.style.setProperty(
			"--primary-text-color",
			"var(--" + color + "-text-color)",
		)
		document.documentElement.style.setProperty(
			"--primary50",
			"var(--" + color + "50)",
		)
		document.documentElement.style.setProperty(
			"--primary50-lightness",
			"var(--" + color + "50-lightness)",
		)
		document.documentElement.style.setProperty(
			"--primary100",
			"var(--" + color + "100)",
		)
		document.documentElement.style.setProperty(
			"--primary100-lightness",
			"var(--" + color + "100-lightness)",
		)
		document.documentElement.style.setProperty(
			"--primary200",
			"var(--" + color + "200)",
		)
		document.documentElement.style.setProperty(
			"--primary200-lightness",
			"var(--" + color + "200-lightness)",
		)
		document.documentElement.style.setProperty(
			"--primary300",
			"var(--" + color + "300)",
		)
		document.documentElement.style.setProperty(
			"--primary300-lightness",
			"var(--" + color + "300-lightness)",
		)
		document.documentElement.style.setProperty(
			"--primary400",
			"var(--" + color + "400)",
		)
		document.documentElement.style.setProperty(
			"--primary400-lightness",
			"var(--" + color + "400-lightness)",
		)
		document.documentElement.style.setProperty(
			"--primary500",
			"var(--" + color + "500)",
		)
		document.documentElement.style.setProperty(
			"--primary500-lightness",
			"var(--" + color + "500-lightness)",
		)
		document.documentElement.style.setProperty(
			"--primary600",
			"var(--" + color + "600)",
		)
		document.documentElement.style.setProperty(
			"--primary600-lightness",
			"var(--" + color + "600-lightness)",
		)
		document.documentElement.style.setProperty(
			"--primary700",
			"var(--" + color + "700)",
		)
		document.documentElement.style.setProperty(
			"--primary700-lightness",
			"var(--" + color + "700-lightness)",
		)
		document.documentElement.style.setProperty(
			"--primary800",
			"var(--" + color + "800)",
		)
		document.documentElement.style.setProperty(
			"--primary800-lightness",
			"var(--" + color + "800-lightness)",
		)
		document.documentElement.style.setProperty(
			"--primary900",
			"var(--" + color + "900)",
		)
		document.documentElement.style.setProperty(
			"--primary900-lightness",
			"var(--" + color + "900-lightness)",
		)
	}
	return (
		<LeftPanelContainer>
			<Flex
				justify="space-between"
				align="center"
				direction="column"
				fullHeight
			>
				<Flex direction="column" align="center">
					<Avatar src="https://avatars.githubusercontent.com/u/10064416?v=4" />
				</Flex>
				<Flex direction="column" align="center" gap={1}>
					{availableColors.map((color) => (
						<div
							onClick={() => changeColor(color)}
							style={{
								background: `linear-gradient(90deg, hsl(
                  var(--${color}-hue),
                  var(--${color}-saturation),
                  calc(var(--${color}-lightness) - 10%)
                ), hsl(
                  var(--${color}-hue),
                  var(--${color}-saturation),
                  calc(var(--${color}-lightness) + 10%)
                ))`,
								boxShadow: `0 0 0 1px hsl(
                  var(--${color}-hue),
                  var(--${color}-saturation),
                  calc(var(--${color}-lightness) - 10%)
                )`,
								border: currentColor === color ? "2px solid white" : "none",
								width: "2rem",
								height: "2rem",
								borderRadius: "50%",
								cursor: "pointer",
							}}
						/>
					))}
					<SIconButton>
						<DoorBackOutlinedIcon />
					</SIconButton>
				</Flex>
			</Flex>
		</LeftPanelContainer>
	)
}

export default LeftPanel
