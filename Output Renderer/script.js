    const fs = require('fs');

    fs.readFile('data.json', 'utf8', (err, data) => {
        if (err) throw err;
        const jsonData = JSON.parse(data);
        displayJsonData(jsonData);
    });


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
    
                if (subKey === "args") {
                    // displayContent += `<li><strong>${subKey}:</strong><ul>`;
    
                    for (const argumentKey in value[subKey]) {
                        const argumentValue = value[subKey][argumentKey];
                        displayContent += `<li><div class=""><strong>${argumentKey}:</strong> ${argumentValue.replace(/\n/g, '<br>')}</div></li>`;
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
        

