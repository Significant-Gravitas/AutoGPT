import { Add, Close, Info } from "@mui/icons-material"
import { AlertTitle } from "@mui/material"
import React from "react"
import { useNavigate } from "react-router"
import Flex from "../../../style/Flex"
import SButton from "../../../style/SButton"
import H2 from "../../UI/atom/H2"
import STextField from "../../UI/atom/STextField"
import SmallText from "../../UI/atom/SmallText"
import { yupResolver } from "@hookform/resolvers/yup"
import Spacer from "../../UI/atom/Spacer"
import SAlert from "../../UI/molecules/SAlert"
import { ColoredIconButton, Container, Modal } from "./InitPage.styled"
import { FieldValues, FormProvider, useForm } from "react-hook-form"
import { schema } from "./InitPage.schema"
import AutoGPTAPI from "../../../api/AutoGPTAPI"
import FTextField from "../../UI/molecules/forms/FTextField"
import FSwitch from "../../UI/molecules/forms/FSwitch"
import Label from "../../UI/atom/Label"

const InitPage = () => {
  const navigate = useNavigate()

  const methods = useForm({
    mode: "onBlur",
    defaultValues: schema.getDefault(),
    resolver: yupResolver(schema),
  })

  const onSubmit = (data: FieldValues) => {
    AutoGPTAPI.createInitData(data)
    navigate("/main")
  }

  return (
    <Container>
      <FormProvider {...methods}>
        <Modal>
          <Flex direction="column" gap={1.5} fullWidth>
            <Flex align="center" justify="space-between">
              <H2 $textColor="grey100">Create a new AI</H2>
              <ColoredIconButton>
                <Close />
              </ColoredIconButton>
            </Flex>
            <Spacer $size={1} />
            <FTextField
              name="ai_name"
              label="Name"
              placeholder="Name your AI"
            />
            <FTextField
              name="ai_role"
              label="Role"
              placeholder="What is your AI for?"
            />
            <Flex direction="column" gap={0.5}>
              <FSwitch name="continuous" label="Continuous mode" />
              <SAlert $color="yellow" $textColor="grey100" severity="info">
                <AlertTitle>Continuous mode</AlertTitle>
                <p>
                  In continuous mode, your AI will be trained every time you add
                  a new goal.
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
              <FTextField
                name="ai_goals[0]"
                label="Goal 1"
                placeholder="Enter your goal number 1"
                noLabel
              />
              <FTextField
                name="ai_goals[1]"
                label="Goal 2"
                placeholder="Enter your goal number 2"
                noLabel
              />
              <FTextField
                name="ai_goals[2]"
                label="Goal 3"
                placeholder="Enter your goal number 3"
                noLabel
              />
              <FTextField
                name="ai_goals[3]"
                label="Goal 4"
                placeholder="Enter your goal number 4"
                noLabel
              />
              <FTextField
                name="ai_goals[4]"
                label="Goal 5"
                placeholder="Enter your goal number 5"
                noLabel
              />
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
                  onClick={methods.handleSubmit(onSubmit)}
                >
                  Start thinking
                </SButton>
              </Flex>
            </Flex>
          </Flex>
        </Modal>
      </FormProvider>
    </Container>
  )
}

export default InitPage
