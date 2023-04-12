import styled, { css } from "styled-components"

const colored = (Component: any) => styled(Component)<{
	$color?: string
	$textColor?: string
	$borderColor?: string
	disabled?: boolean
}>`
  ${({ $color = "blue", disabled = false, $textColor }) => {
		const color = $color?.match(/([a-z]+)([0-9]{3})?/)

		// set default text color to white if lightness is high
		const textColor = color && parseInt(color[2]) > 500 ? "white" : "black"
		let finalCss = ``
		if (!$textColor) {
			if (color && !color[2]) {
				finalCss += `
            --text-color: var(--${color[1]}-text-color);
          `
			} else {
				finalCss += `
            --text-color: ${textColor};
          `
			}
		}
		if (color && color[2]) {
			finalCss += `
        --color-hue: var(--${color[1]}-hue);
        --color-saturation: var(--${color[1]}-saturation);
        --color-lightness: var(--${$color}-lightness);
      `
		} else {
			finalCss += `
      --color-hue: var(--${$color}-hue);
      --color-saturation: var(--${$color}-saturation);
      --color-lightness: var(--${$color}-lightness);
    `
		}
		return (finalCss += `
      --color: hsl(
        var(--color-hue),
        var(--color-saturation),
        var(--color-lightness)
      );
      --color-hover: hsl(
        var(--color-hue),
        var(--color-saturation),
        calc(var(--color-lightness) + 5%)
      );
      --color-active: hsl(
        var(--color-hue),
        var(--color-saturation),
        calc(var(--color-lightness) - 5%)
      );
      --shadow-color: var(--color);
      transition: all 0.3s ease-in-out;
    `)
	}}
  ${({ $textColor }) => {
		const textColor = $textColor?.match(/([a-z]+)([0-9]{3})?/)
		let finalCss = ``
		if (textColor) {
			return (finalCss += `
        --text-color-hue: var(--${textColor[1]}-hue);
        --text-color-saturation: var(--${textColor[1]}-saturation);
        --text-color-lightness: var(--${$textColor}-lightness);
        --text-color: hsl(
          var(--text-color-hue),
          var(--text-color-saturation),
          var(--text-color-lightness)
        );
      `)
		} else {
			return ``
		}
	}}

  ${({ $borderColor }) => {
		const borderColor = $borderColor?.match(/([a-z]+)([0-9]{3})?/)
		let finalCss = ``
		if (borderColor) {
			return (finalCss += `
        --border-color-hue: var(--${borderColor[1]}-hue);
        --border-color-saturation: var(--${borderColor[1]}-saturation);
        --border-color-lightness: var(--${$borderColor}-lightness);
        --border-color: hsl(
          var(--border-color-hue),
          var(--border-color-saturation),
          var(--border-color-lightness)
        );
      `)
		} else {
			return ``
		}
	}}
`

export default colored
