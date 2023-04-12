import Flex from "@/style/Flex"
import { Switch } from "@mui/material"
import { useController, useFormContext } from "react-hook-form"
import styled, { css } from "styled-components"
import Label from "../../atom/Label"

export const StyledSwitch = styled(Switch)<{ $active?: boolean }>`
  & * {
    color: var(--primary) !important;
  }
  & .MuiSwitch-track {
    background-color: var(--primary100) !important;
    opacity: 0.2 !important;
    ${({ $active }) =>
			$active &&
			css`
        opacity: 0.7 !important;
        background-color: var(--primary) !important;
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
		<Flex gap={0.5} align="center">
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
