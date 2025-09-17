const express = require('express');
const cors = require('cors');
const axios = require('axios');
const bodyParser = require('body-parser');

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '12mb' }));

const ML_API = process.env.ML_API || 'http://localhost:5000/predict';
const ML_MULTI_API = process.env.ML_MULTI_API || 'http://localhost:5001/predict';

app.post('/api/predict', async (req, res) => {
  const { image } = req.body;
  if (!image) return res.status(400).json({ success: false, error: 'No image sent' });
  try {
    const response = await axios.post(ML_API, { image: image }, { timeout: 15000 });
    res.json(response.data);
  } catch (err) {
    const msg = err.response ? err.response.data : err.message;
    res.status(500).json({ success: false, error: msg });
  }
});

app.post('/api/multipredict', async (req, res) => {
  const { image } = req.body;
  if (!image) return res.status(400).json({ success: false, error: 'No image sent' });
  try {
    const response = await axios.post(ML_MULTI_API, { image: image }, { timeout: 20000 });
    res.json(response.data);
  } catch (err) {
    const msg = err.response ? err.response.data : err.message;
    res.status(500).json({ success: false, error: msg });
  }
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`Node backend listening on port ${PORT}`));
