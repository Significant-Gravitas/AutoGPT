// Define two stacks for left and right
const leftStack = [];
const rightStack = [];
let lastAction = false;

// Define an array of colors to choose from
const colors = ["rgb(80,38, 56)", "#E0BBE4", "#957DAD", "#D291BC", "#FEC8D8", "#FFDFD3"];
// Keep track of assigned colors
const assignedColors = new Map();

const chat = document.getElementById("messages-display");

setInterval(fetchData, 1000); // Update the data every 1 second (1000 ms)

function fetchData() {
    let prevData = null;

    //read from file


    fetch('message.json')
        .then(response => response.json())
        .then(data => {
            // update UI here
            if (data === prevData) {
                return;
            }
            prevData = data;
            renderProfile(data)
            if (lastAction) {
                chat.lastElementChild.remove();
            }
            switch (data.event) {
                case 'message':
                    lastAction = false;
                    renderMessage(data);
                    break;
                case 'working':
                    lastAction = true;
                    renderAction(data);
                    break;
                default:
                    break;
            }
        });
}

function renderProfile(data) {
    let planTextField;
    let rhs = document.getElementById("chat-rhs");
    rhs.innerHTML = "";

    // Create a new div element for the profile card
    var profileCardDiv = document.createElement("div");
    profileCardDiv.className = "profile-card";

    // Create a div element for the profile
    var profileDiv = document.createElement("div");
    profileDiv.className = "profile";
    profileCardDiv.appendChild(profileDiv);

    // Create a div element for the profile picture
    var profilePicDiv = document.createElement("div");
    profilePicDiv.className = "profile-pic";
    profileDiv.appendChild(profilePicDiv);

    // Create a div element for the profile text
    var profileTextDiv = document.createElement("div");
    profileTextDiv.className = "profile-text";
    profileDiv.appendChild(profileTextDiv);

    // Create a h2 element for the profile name
    var profileNameH2 = document.createElement("h2");
    profileNameH2.className = "profile-name";
    profileNameH2.textContent = data.name;
    profileTextDiv.appendChild(profileNameH2);

    // Create a div element for the profile bio
    var profileBioDiv = document.createElement("div");
    profileBioDiv.className = "profile-bio";
    profileBioDiv.innerHTML = "<i>" + data.role + "</i>";
    profileTextDiv.appendChild(profileBioDiv);

    // Create a text-field element for the goals
    // var goalsTextField = createListElement("goal", data.goals, false);
    // profileCardDiv.appendChild(goalsTextField);

    // Create a text field for thought if it exists in the data
    if (data.thought) {
        var thoughtTextField = createTextField("thought", data.thought);
        profileCardDiv.appendChild(thoughtTextField);
    }

    if (data.reasoning) {
        var reasoningTextField = createTextField("reasoning", data.reasoning);
        profileCardDiv.appendChild(reasoningTextField);
    }

    if (data.plan) {
        const [ifOrdered, result] = splitAndTrimString(data.plan);
        if (ifOrdered) {
            planTextField = createListElement("plan", result, true);
        } else {
            planTextField = createTextField("plan", result);
        }
        profileCardDiv.appendChild(planTextField);
    }

    if (data.criticism) {
        var criticismTextField = createTextField("criticism", data.criticism);
        profileCardDiv.appendChild(criticismTextField);
    }

    rhs.appendChild(profileCardDiv);
}

function createListElement(name, list, ifOrdered) {
    // Create a new div element
    var div = document.createElement("div");
    div.className = "text-field";

    // Create a new label element
    var label = document.createElement("label");
    label.htmlFor = name;
    label.textContent = name.charAt(0).toUpperCase() + name.slice(1);
    div.appendChild(label);

    // Create a new unordered or ordered list element based on ifOrdered parameter
    var ul = ifOrdered ? document.createElement("ol") : document.createElement("ul");
    ul.id = name;

    // Loop through the list array and create list items for each item
    for (var i = 0; i < list.length; i++) {
        var li = document.createElement("li");
        li.textContent = list[i];
        ul.appendChild(li);
    }

    div.appendChild(ul);

    return div;
}

function createTextField(name, textContent) {
    var textField = document.createElement("div");
    textField.className = "text-field";
    var label = document.createElement("label");
    label.htmlFor = name;
    label.textContent = name.charAt(0).toUpperCase() + name.slice(1);
    var paragraph = document.createElement("p");
    paragraph.id = name;
    paragraph.name = name;
    paragraph.textContent = textContent;
    textField.appendChild(label);
    textField.appendChild(paragraph);
    return textField;
}

function splitAndTrimString(str) {
    // Check if the input string contains a hyphen
    if (str.includes("-")) {
        // Split the string into substrings using hyphen as delimiter
        const substrings = str.split("-");

        // Trim each substring
        const trimmedSubstrings = substrings.map(substring => substring.trim());

        // Return the array of trimmed substrings
        return [true, trimmedSubstrings];
    }
    // If the input string does not contain a hyphen, return an array with the original string
    return [false, [str.trim()]];
}

function renderAction(data) {
    const messageGroup = document.createElement('div');
    messageGroup.classList.add('message-group-received');

    const messageElement = document.createElement('div');
    messageElement.classList.add('message-received');

    const messageText = document.createElement('div');
    messageText.classList.add('message-received-text');

    const messageIcon = document.createElement('div');
    messageIcon.classList.add('message-received-icon');
    messageIcon.innerHTML = "<i style='opacity: 0.5;'>" + data.name + " " + data.type + "</i>";

    const loader = document.createElement('div');
    loader.className = 'loader';

    const dotL = document.createElement('div');
    dotL.className = 'dot left';

    const dotC = document.createElement('div');
    dotC.className = 'dot center';

    const dotR = document.createElement('div');
    dotR.className = 'dot right';

    loader.append(dotL);
    loader.append(dotC);
    loader.append(dotR);
    messageText.append(loader);
    messageElement.append(messageText);
    messageElement.append(messageIcon);
    messageGroup.append(messageElement);
    messageGroup.dataset.type = 'loader';
    chat.append(messageGroup);
    chat.scrollTop = chat.scrollHeight;
}

function renderMessage(data) {
    let [color, direction] = getSenderInfo(data.name);
    // TODO: what is the image source
    imagesrc = "./images/happy_emoji.png"
    addMessageElement(data.message, imagesrc, direction, color, data.name);
    chat.scrollTop = chat.scrollHeight;
}

function getSenderInfo(name) {
    // Check if the sender's name has been seen before
    if (assignedColors.has(name)) {
        // If sender's name is found in assigned colors, return the assigned color and stack
        const assignedColor = assignedColors.get(name);
        return [assignedColor, leftStack.findIndex(entry => entry === name) !== -1];
    } else {
        // If sender's name is not found in assigned colors, assign a unique color
        const availableColors = colors.filter(color => !Array.from(assignedColors.values()).includes(color));
        const assignedColor = availableColors.length > 0 ? availableColors[0] : null;
        if (assignedColor) {
            assignedColors.set(name, assignedColor);
            // Assign the sender's name, message, and assigned color to the smaller stack
            if (leftStack.length <= rightStack.length) {
                leftStack.push(name);
                return [assignedColor, true];
            } else {
                rightStack.push(name);
                return [assignedColor, false];
            }
        } else {
            // If all colors are already assigned, return null
            return null;
        }
    }
}

function addMessageElement(message, icon, direction, color, name) {
    let container = chat.lastElementChild;
    let messageContainer;
    let newGroup = false
    if (container !== null && container.getAttribute('data-name') === name) {
        messageContainer = chat.lastElementChild.lastElementChild;
    } else {
        // Create the main container div
        newGroup = true
        container = document.createElement('div');
        container.setAttribute('data-name', name);

        if (direction) {
            container.className = 'message-group-received';
        } else {
            container.className = 'message-group-sent';
        }
        // Create the first child div for the image
        const imageContainer = document.createElement('div');
        imageContainer.className = 'icon-container';
        container.appendChild(imageContainer);

        // Create the image element and set its source
        const image = document.createElement('img');
        image.src = icon;
        imageContainer.appendChild(image);

        const iconName = document.createElement('div');
        iconName.className = 'icon-name';
        iconName.textContent = name;
        imageContainer.appendChild(iconName);

        // Create the second child div for the message
        messageContainer = document.createElement('div');
        container.appendChild(messageContainer);
    }

    // Create the inner div for the message text
    const messageTextContainer = document.createElement('div');

    // Set the class of the inner div based on the direction
    if (direction) {
        messageTextContainer.className = 'message-received-text';
    } else {
        messageTextContainer.className = 'message-sent-text';
    }

    // Set the text content of the inner div to the input message
    messageTextContainer.textContent = message;
    messageTextContainer.style.backgroundColor = color;

    // Create the message element based on the direction
    if (direction) {
        const messageReceived = document.createElement('div');
        messageReceived.className = 'message-received';
        messageReceived.appendChild(messageTextContainer);
        messageContainer.appendChild(messageReceived);
    } else {
        const messageSent = document.createElement('div');
        messageSent.className = 'message-sent';
        messageSent.appendChild(messageTextContainer);
        messageContainer.appendChild(messageSent);
    }

    if (newGroup)
        chat.appendChild(container);

}
