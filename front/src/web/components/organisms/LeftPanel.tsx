import React from "react"
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
        <SIconButton>
          <DoorBackOutlinedIcon />
        </SIconButton>
      </Flex>
    </LeftPanelContainer>
  )
}

export default LeftPanel
