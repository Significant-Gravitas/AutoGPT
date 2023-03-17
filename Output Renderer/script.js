const jsonData = {
    "command":
    {
    "name": "Commit to Long Term Memory",
    "arguments":
    {
    "string": "1. AI-based Photo Editing Apps\n2. AI-Based Writing Tool\n3. Advertising Software\n4. AI Marketing Agency\n5. Recruitment Business App\n6. AI-powered Cybersecurity App\n7. Healthcare Startup\n8. Medical Equipment Business"
    }
    },
    "Thoughts":
    {
    "text": "Storing the ranking of potential AI-based businesses in Long Term Memory",
    "reasoning": "To remember the ranked list and use it to guide the decision-making process in choosing the most suitable business opportunity",
    "current long-term plan": "- Research potential businesses\n- Choose a suitable business\n- Develop and manage the business autonomously",
    "critisism": "None at the moment"
    }
};

function displayJsonData() {
    const jsonDisplay = document.getElementById('json-display');
    let displayContent = '';

    for (const key in jsonData) {
        displayContent += `<h2>${key}</h2><ul>`;
        const value = jsonData[key];

        for (const subKey in value) {
            let subValue = value[subKey];

            if (typeof subValue === 'string') {
                subValue = subValue.replace(/\n/g, '<br>');
            }

            if (subKey === "arguments") {
                displayContent += `<li><strong>${subKey}:</strong><ul><li><div class="argument-box"><strong>string:</strong> ${value[subKey].string.replace(/\n/g, '<br>')}</div></li></ul></li>`;
            } else {
                displayContent += `<li><strong>${subKey}:</strong> ${subValue}</li>`;
            }
        }

        displayContent += '</ul>';
    }

    jsonDisplay.innerHTML = displayContent;
}

displayJsonData();
