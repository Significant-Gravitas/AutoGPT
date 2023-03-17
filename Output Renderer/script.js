const jsonData = {
    "command": {
    "name": "Google Search",
    "arguments": {
    "search": "simple and profitable online business ideas"
    }
    },
    "thoughts": {
    "text": "I will search for simple and profitable online business ideas to begin my entrepreneurship journey.",
    "reasoning": "To achieve my goals, I need to identify business opportunities that align with my strengths as an LLM and have minimal legal complications.",
    "current long-term plan": "- Search for business ideas\n- Evaluate and select ideas\n- Develop and implement chosen business strategy\n- Continuously refine strategies based on market trends and performance metrics",
    "critisism": "I must ensure that the chosen business idea is both simple and legal, while also considering scalability and profitability."
    }
    };

    function displayJsonData() {
        const jsonDisplay = document.getElementById('json-display');
        let displayContent = '';
    
        for (const key in jsonData) {
            displayContent += `<h2>${key}:</h2><br><ul>`;
            const value = jsonData[key];
    
            for (const subKey in value) {
                let subValue = value[subKey];
    
                if (typeof subValue === 'string') {
                    subValue = subValue.replace(/\n/g, '<br>');
                }
    
                if (subKey === "arguments") {
                    displayContent += `<li><strong>${subKey}:</strong><ul>`;
    
                    for (const argumentKey in value[subKey]) {
                        const argumentValue = value[subKey][argumentKey];
                        displayContent += `<li><div class="argument-box"><strong>${argumentKey}:</strong> ${argumentValue.replace(/\n/g, '<br>')}</div></li>`;
                    }
    
                    displayContent += '</ul></li>';
                } else {
                    displayContent += `<li><strong>${subKey}:</strong> ${subValue}</li>`;
                }
            }
    
            displayContent += '</ul>';
        }
    
        jsonDisplay.innerHTML = `
            <div class="json-display-content">
                ${displayContent}
            </div>
            <div class="twitter-handles">
                    <img src="en_gpt.jpg" alt="En_GPT" class="twitter-profile-pic">
                <a href="https://twitter.com/En_GPT"> @En_GPT</a>     
                    <img src="SigGravitas.jpg" alt="SigGravitas" class="twitter-profile-pic">
                <a href="https://twitter.com/SigGravitas">@SigGravitas</a>
            </div>
        `;
    }
        
    displayJsonData();
