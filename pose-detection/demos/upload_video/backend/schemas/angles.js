import { mongoose } from 'mongoose';

const advanceSchema = new mongoose.Schema({
    frontKnee: Number,
    elbow: Number,
    torse: Number,
    backKnee: Number,
});

const retreatSchema = new mongoose.Schema({
    frontKnee: Number,
    elbow: Number,
    torse: Number,
    backKnee: Number,
  });

  const lungeSchema = new mongoose.Schema({
    frontKnee: Number,
    elbow: Number,
    torse: Number,
    backKnee: Number,
  });

  const enGardeSchema = new mongoose.Schema({
    frontKnee: Number,
    elbow: Number,
    torse: Number,
    backKnee: Number,
  });

const advance = mongoose.model('advance', advanceSchema);
const retreat = mongoose.model('retreat', retreatSchema);
const lunge = mongoose.model('lunge', lungeSchema);
const enGarde = mongoose.model('enGarde', enGardeSchema);

module.exports = advance;
module.exports = retreat;
module.exports = lunge;
module.exports = enGarde;
