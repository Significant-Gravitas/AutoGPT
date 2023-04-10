import React, { useState } from "react"
import styled from "styled-components"
import Flex from "../../../../style/Flex"
import Label from "../../atom/Label"
import { Switch } from "@mui/material"
import { useController, useFormContext } from "react-hook-form"

export const StyledSwitch = styled(Switch)<{ $active?: boolean }>`
  & * {
    color: var(--yellow) !important;
  }
  & .MuiSwitch-track {
    background-color: var(--yellow100) !important;
    ${({ $active }) =>
      $active &&
      `
      background-color: var(--yellow) !important;
    `}
`

interface IFSwitchProps {
  name: string
  label?: string
}

const FSwitch = ({ name, label }: IFSwitchProps) => {
  const { control } = useFormContext()

  const {
    field: { ref, ...inputProps },
    fieldState: { error },
  } = useController({
    name,
    control,
    defaultValue: "",
  })
  return (
    <Flex direction="column" gap={0.5}>
      <Label>{label ?? name}</Label>
      <StyledSwitch
        disableRipple
        $active={inputProps.value}
        inputRef={ref}
        {...inputProps}
      />
    </Flex>
  )
}

export default FSwitch
