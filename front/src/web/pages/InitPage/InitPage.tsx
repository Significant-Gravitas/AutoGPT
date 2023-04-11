import { yupResolver } from "@hookform/resolvers/yup"
import { Add, Close, Info } from "@mui/icons-material"
import { AlertTitle } from "@mui/material"
import { FieldValues, FormProvider, useForm, useWatch } from "react-hook-form"
import { useNavigate } from "react-router"
import AutoGPTAPI from "../../api/AutoGPTAPI"
import H2 from "../../components/atom/H2"
import Label from "../../components/atom/Label"
import SmallText from "../../components/atom/SmallText"
import Spacer from "../../components/atom/Spacer"
import SAlert from "../../components/molecules/SAlert"
import FSwitch from "../../components/molecules/forms/FSwitch"
import FTextField from "../../components/molecules/forms/FTextField"
import Flex from "../../style/Flex"
import SButton from "../../style/SButton"
import { schema } from "./InitPage.schema"
import { ColoredIconButton, Container, Modal, OnOff } from "./InitPage.styled"
import { addAiHistory } from "../../redux/data/dataReducer"
import { useDispatch } from "react-redux"
import { v4 as uuidv4 } from "uuid"
const InitPage = () => {
  const navigate = useNavigate()
  const dispatch = useDispatch()
  const methods = useForm({
    mode: "onBlur",
    defaultValues: schema.getDefault(),
    resolver: yupResolver(schema),
  })

  const onSubmit = (data: FieldValues) => {
    AutoGPTAPI.createInitData(data)
    const newId = uuidv4()
    dispatch(
      addAiHistory({
        id: newId,
        agents: [],
        name: data.ai_name,
        role: data.ai_role,
        goals: data.goals,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        answers: [],
      }),
    )
    navigate(`/main/${newId}`)
  }

  const continuous = useWatch({
    control: methods.control,
    name: "continuous",
  })

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
              <Flex gap={0.5} align="center">
                <FSwitch name="continuous" label="Continuous mode :" />
                <OnOff>{continuous ? "ON" : "OFF"}</OnOff>
              </Flex>
              {continuous && (
                <SAlert $color="yellow" $textColor="grey100" severity="info">
                  <AlertTitle>Continuous mode</AlertTitle>
                  <p>
                    In continuous mode, your AI will be trained every time you
                    add a new goal.
                  </p>
                </SAlert>
              )}
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
