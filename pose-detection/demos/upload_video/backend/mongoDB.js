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
      
      // API endpoint to get all courses
      app.get('/api/angles', async (req, res) => {
        try {
          await client.connect();
          const collection = client.db('Garde').collection('Garde-User-Angles');
          const angles = await collection.find({}).toArray();
          res.json(angles);
        } catch (error) {
          res.status(500).json({ error: error.toString() });
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
              throw new Error('Failed to insert new course.');
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
            // const objectId = new ObjectId(id);
            const result = await collection.deleteOne({ _id: id });

          } catch(error) {
            res.status(500).json({ error: error.toString() });
          }
        });
    
  } catch (err) {
    console.error("Connection to MongoDB failed", err);
  }
}

run().catch(console.dir);
