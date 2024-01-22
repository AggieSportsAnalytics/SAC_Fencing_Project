const { OpenAI } = require('openai')

const openai = new OpenAI({ apiKey: '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' });

const ideal_en_guarde = {"name":"En-Guarde","elbow_left":"96","hip_left":"117","knee_left":"121","elbow_right":"2","hip_right":"170","knee_right":"160"};
const ideal_advance = {"name":"Advance","elbow_left":"87","hip_left":"126","knee_left":"132","elbow_right":"36","hip_right":"170","knee_right":"160"};
const ideal_retreat = {"name":"Retreat","elbow_left":"90","hip_left":"127","knee_left":"144","elbow_right":"8","hip_right":"172","knee_right":"170"};
const ideal_lunge = {"name":"Lunge","elbow_left":"178","hip_left":"84","knee_left":"110","elbow_right":"170","hip_right":"151","knee_right":"165"};

function toJSON(user_angles) {
    return {"name":`${user_angles.name}`,"elbow_left":`${user_angles.elbow_left}`,"hip_left":`${user_angles.hip_left}`,"knee_left":`${user_angles.knee_left}`,"elbow_right":`${user_angles.elbow_right}`,"hip_right":`${user_angles.hip_right}`,"knee_right":`${user_angles.knee_right}`};
}

async function compareAngles(user_angles) {

    let userAngles = toJSON(user_angles);
    let comparison;

    if(user_angles.name == "En-Guarde") {
        comparison = ideal_en_guarde;
    }
    else if(user_angles.name == "Advance") {
        comparison = ideal_advance;
    }
    else if(user_angles.name == "Retreat") {
        comparison = ideal_retreat;
    }
    else if(user_angles.name == "Lunge") {
        comparison = ideal_lunge;
    }

    let dataComp = `User's angles: ${JSON.stringify(userAngles)}, ideal angles: ${JSON.stringify(comparison)}.`;
    let query = `Please compare the user's angles with the ideal angles for the ${user_angles.name} position and provide a detailed analysis.`;

    const messages = [
        {"role": "system", "content": dataComp},
        {"role": "user", "content": query}
    ];

    const completion = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: messages
    });

    // return completion.choices[0].message.content;
    console.log(completion.choices[0].message.content);
}

testAngle = {
    name: "En-Guarde", 
    elbow_left: 123,
    hip_left: 150,
    knee_left: 90, 
    elbow_right: 0,
    hip_right: 200,
    knee_right: 100 
};

compareAngles(testAngle);

// module.exports = { compareAngles };