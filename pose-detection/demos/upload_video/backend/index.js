const express = require('express');
const cors = require('cors');
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
require('dotenv').config({ path: '../../../../.env' });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const fs = require('fs').promises;

async function loadJson(filePath) {
  try {
    const data = await fs.readFile(filePath, 'utf8');
    return JSON.parse(data); 
  } catch (error) {
    console.error('Error reading or parsing the file:', error);
    return null; 
  }
}

function toJSON(user_angles) {
    return {"name":`${user_angles.name}`,"elbow_left":`${user_angles.elbow_left}`,"hip_left":`${user_angles.hip_left}`,"knee_left":`${user_angles.knee_left}`,"elbow_right":`${user_angles.elbow_right}`,"hip_right":`${user_angles.hip_right}`,"knee_right":`${user_angles.knee_right}`};
}

async function compareAngles(user_angles) {
    
    ideal_angles = await loadJson('ideal_angles.json');
    let ideal_en_guarde = ideal_angles[0];
    let ideal_advance = ideal_angles[1];
    let ideal_retreat = ideal_angles[2];
    let ideal_lunge = ideal_angles[3];

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