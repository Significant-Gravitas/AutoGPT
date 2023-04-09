import React from "react"
import {
  ColoredIconButton,
  Container,
  Label,
  Modal,
  StyledSwitch,
} from "./InitPage.styled"
import SButton from "../../../style/SButton"
import { Add, Close, Info } from "@mui/icons-material"
import Flex from "../../../style/Flex"
import { useNavigate } from "react-router"
import STextField from "../../UI/atom/STextField"
import H2 from "../../UI/atom/H2"
import { Alert, AlertTitle, IconButton, Slider, Switch } from "@mui/material"
import colored from "../../../style/colored"
import styled from "styled-components"
import Spacer from "../../UI/atom/Spacer"
import SmallText from "../../UI/atom/SmallText"
import SAlert from "../../UI/molecules/SAlert"

const InitPage = () => {
  const navigate = useNavigate()
  const [isContinuous, setIsContinuous] = React.useState(false)
  return (
    <Container>
      <Modal>
        <Flex direction="column" gap={1.5} fullWidth>
          <Flex align="center" justify="space-between">
            <H2 $textColor="grey100">Create a new AI</H2>
            <ColoredIconButton>
              <Close />
            </ColoredIconButton>
          </Flex>
          <Spacer $size={1} />
          <Flex direction="column" gap={0.5}>
            <Label>Name</Label>
            <STextField placeholder="Name your AI" variant="outlined" />
          </Flex>
          <Flex direction="column" gap={0.5}>
            <Label>Role</Label>
            <STextField placeholder="What is your AI for?" variant="outlined" />
          </Flex>
          <Flex direction="column" gap={0.5}>
            <Label>Continuous mode</Label>
            <StyledSwitch
              disableRipple
              onChange={() => {
                setIsContinuous(!isContinuous)
              }}
              value={isContinuous}
              $active={isContinuous}
            />
            <SAlert $color="yellow" $textColor="grey100" severity="info">
              <AlertTitle>Continuous mode</AlertTitle>
              <p>
                In continuous mode, your AI will be trained every time you add a
                new goal.
              </p>
            </SAlert>
          </Flex>

          <Flex direction="column" gap={0.5}>
            <Label>Goals</Label>
            <SmallText $textColor="grey400">
              <Flex align="center" gap={0.5}>
                <Info fontSize="small" />
                Enter up to 5 goals for your AI
              </Flex>
            </SmallText>
            <STextField placeholder="Goal 1" variant="outlined" />
            <STextField placeholder="Goal 2" variant="outlined" />
            <STextField placeholder="Goal 3" variant="outlined" />
            <STextField placeholder="Goal 4" variant="outlined" />
            <STextField placeholder="Goal 5" variant="outlined" />
          </Flex>
          <Flex fullWidth align="flex-end" justify="space-between">
            <div />
            <Flex gap={2}>
              <SButton variant="text" $color="grey100">
                Cancel
              </SButton>
              <SButton
                $color="yellow"
                variant="contained"
                startIcon={<Add />}
                onClick={() => {
                  navigate("/main")
                }}
              >
                Start thinking
              </SButton>
            </Flex>
          </Flex>
        </Flex>
      </Modal>
    </Container>
  )
}

export default InitPage
