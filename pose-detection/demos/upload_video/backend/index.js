const express = require('express');
const cors = require('cors');
// const openAI = require('./openAI');
const { MongoClient, ServerApiVersion, ObjectId } = require('mongodb');

const app = express();
app.use(express.json()); // Middleware to parse JSON bodies
app.use(cors()); // Enable CORS

const uri = "mongodb+srv://asa_admin:gardeasa_admin@en-garde.d5nem9m.mongodb.net/?retryWrites=true&w=majority"
const client = new MongoClient(uri, {
  serverApi: ServerApiVersion.v1,
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

// Connect to MongoDB once when the server starts
async function run() {
  try {
    await client.connect();
    console.log("Connected successfully to MongoDB");

    // Start the server after successful MongoDB connection
    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });

    // API endpoint to get all angles
    // app.get('/api/angles', async (req, res) => {
    //   try {
    //     await client.connect();
    //     const collection = client.db('Garde').collection('Garde-User-Angles');
    //     const angles = await collection.find({}).toArray();
    //     console.log("uploading angles");
    //     res.json(angles);
    //     compareAngles(angles[angles.length-1]);
    //   } catch (error) {
    //     res.status(500).json({ error: error.toString() });
    //   } 
    // });

    app.post('/api/gpt', async (req, res) => {
      try {
          const result = await compareAngles(req.body);
          // const result = "";
          res.json({ message: result });
      } catch (error) {
          res.status(500).send('Error processing request');
      }
    });
    
    app.post('/api/angles', async (req, res) => {
      try {
        await client.connect();
        const collection = client.db('Garde').collection('Garde-User-Angles');
        const angles = req.body; // The course data sent in the request body
        const result = await collection.insertOne(angles);
        
        // If you want to return the entire new course document, use the 'insertedId' to fetch it.
        // The inserted document is not returned directly in the result of `insertOne()`.
        if (result.insertedId) {
          const newCourse = await collection.findOne({ _id: result.insertedId });
          res.status(201).json(newCourse);
        } else {
          throw new Error('Failed to insert new angles.');
        }
    
      } catch (error) {
        console.error(error);
        res.status(500).json({ error: error.message });
      }
    });

    app.delete('/api/angles/:id', async (req, res) => {
      try {
        await client.connect();
        const collection = client.db('Garde').collection('Garde-User-Angles');
        const id = req.params.id;
        const objectId = new ObjectId(id);
        const result = await collection.deleteOne({ _id: objectId });

      } catch(error) {
        res.status(500).json({ error: error.toString() });
      }
    });
  
  } catch (err) {
    console.error("Connection to MongoDB failed", err);
  }
}

run().catch(console.dir);

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
        messages: messages,
        // stream: true,
    });
    // const stream = OpenAIStream(completion);
    // return completion.choices[0].message.content;
    // console.log(completion.choices[0].message.content);

    return completion.choices[0].message.content;
    // return new StreamingTextResponse(stream);
}