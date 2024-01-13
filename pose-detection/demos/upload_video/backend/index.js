const { MongoClient, ServerApiVersion } = require('mongodb');
const uri = "mongodb+srv://asa_admin:gardeasa_admin@en-garde.d5nem9m.mongodb.net/?retryWrites=true&w=majority"

const client = new MongoClient(uri, {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: true,
    deprecationErrors: true,
  }
});

async function run () {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log("Successfully connected to the mongoDB server!");
  } finally {
    await client.close();
  }
}

async function addUserAngle(document) {
  try {
    await client.connect();

    const db = client.db("Garde");
    const collection = db.collection("Garde-User-Angles");

    const insertResult = await collection.insertOne(document);
    console.log('Inserted document:', insertResult.insertedId);

  } finally {
    await client.close();
  }
}

//Test entry works
// const userAngles = {
//   _id: 1, 
//   name: 'Test Entry', 
//   elbow_left: 0,
//   hip_left: 0,
//   knee_left: 0, 
//   elbow_right: 0,
//   hip_right: 0,
//   knee_right: 0 
// };
// addUserAngle(userAngles).catch(console.dir);

// run().catch(console.dir);