const express = require('express');
const app = express();
const multer = require('multer');

const storage = multer.diskStorage({
  destination: function(req, file, cb){
    cb(null, 'files');
  },
  filename: function(req,file,cb){
    cb(null, file.originalname);
  }
})

const upload = multer({storage});

const tf = require('@tensorflow/tfjs-node');
const classes = require('./classes.json');
const fs = require('fs').promises;
const fs2 = require('fs');

const loadModel = async () => {
  const modelFile = tf.io.fileSystem('./food_detection_model/model.json');
  const model = await tf.loadGraphModel(modelFile);
  return new Promise((resolve) => {
    resolve(model);
  });
};

let model;

const preProcess = async (path) => {
  try {
    const buf = await fs.readFile(path);
    const preprocessedData = tf.tidy(() => {
      const data = tf.node.decodeImage(buf);
      const resizedData = tf.image.resizeBilinear(data, [512, 512]);
      const normalizedData = tf.div(resizedData, 255.0);
      return normalizedData.expandDims(0).transpose([0, 3, 1, 2]);
    });

    return new Promise((resolve) => {
      resolve(preprocessedData);
    });
  } catch (err) {
    console.error(err);
  }

  return null;
};

const postProcess = async (res) => {
  const boxes = tf.tidy(() => {
    const h = res.slice([0, 3, 0], [1, 1, -1]);
    const w = res.slice([0, 2, 0], [1, 1, -1]);
    const y1 = res.slice([0, 1, 0], [1, 1, -1]).sub(tf.div(h, 2));
    const x1 = res.slice([0, 0, 0], [1, 1, -1]).sub(tf.div(w, 2));

    return tf
      .concat([y1, x1, y1.add(h), x1.add(w)])
      .transpose()
      .squeeze();
  });

  const [scores, labels] = tf.tidy(() => {
    const confClasses = res.slice([0, 4, 0], [1, classes.length, -1]).transpose().squeeze();
    return [confClasses.max(1), confClasses.argMax(1)];
  });

  const nonMaxSupression = await tf.image.nonMaxSuppressionAsync(boxes, scores, 5, 0.7, 0.5);
  const detectedObject = tf.tidy(() => {
    const uniques = tf.unique(labels.gather(nonMaxSupression));
    return uniques.values.arraySync();
  });
  const stringClasses = detectedObject.map((idx) => classes[idx]);


  console.log(
    `------ Hasil Prediksi ------\nTerdapat ${detectedObject.length} objek diantaranya : ${stringClasses.join(', ')}`
  );

  tf.dispose([boxes, scores, labels, nonMaxSupression]);
  return stringClasses;
};

const predict = async (path) => {
  const gambar = await preProcess(path);

  if (gambar === null) {
    console.log('Gambar Tidak ditemukan');

    // error code or something
    return;
  }

  const res = await model.predict(gambar);
  const temp = await postProcess(res);
  tf.dispose([res, gambar]);

  return temp;

};

app.get('/',async(req,res,next)=>{
  try {
    res.status(200).json({
      message: "Hello World!"
    })
  } catch (error) {
    console.log(error.message);
    res.status(error.status || 500).json({
      status: "Failed to upload files",
      message: error.message
    })
  }
})

app.post('/file', upload.single('image'), async(req,res,next)=>{
  try {
    const result = await predict('files/test-image.jpg');
    res.status(200).json({
      status: "Success",
      result
    })
  } catch (error) {
    console.log(error.message);
    res.status(error.status || 500).json({
      status: "Failed to upload files",
      message: error.message
    })
  }
})

loadModel()
.then((result)=>{
  model = result
  app.listen(5000, ()=>{
    console.log("Server is listening on PORT 5000");
  })
})