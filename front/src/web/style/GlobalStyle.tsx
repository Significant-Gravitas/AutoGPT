import { createGlobalStyle } from "styled-components"
import colors from "./fondation/colors"

const GlobalStyle = createGlobalStyle`

  @font-face {
    font-family: 'Manrope';
    src: local('Manrope'), local('Manrope'),
          url('/fonts/Manrope-Regular.ttf') format('truetype');
    font-weight: 400;
    font-style: normal;
  }

  @font-face {
    font-family: 'Manrope';
    src: local('Manrope'), local('Manrope'),
        url('/fonts/Manrope-Medium.ttf') format('truetype');
    font-weight: 500;
    font-style: normal;
    }

    @font-face {
    font-family: 'Manrope';
    src: local('Manrope'), local('Manrope'),
            url('/fonts/Manrope-SemiBold.ttf') format('truetype');
    font-weight: 600;
    font-style: normal;
    }

  @font-face {
    font-family: 'Manrope';
    src: local('Manrope'), local('Manrope'),

          url('/fonts/Manrope-Bold.ttf') format('truetype');
    font-weight: 700;
    font-style: normal;
  }

  * {
    font-family: 'Manrope', sans-serif !important;
    box-sizing: border-box !important;
  }

 :root {
    ${colors}
    --ai-list-width: 400px;
  }
body {
    margin: 0;
    padding: 0;
}
* {
    box-sizing: border-box;
}
`
export default GlobalStyle
