import React, { useRef, useEffect, useState } from 'react';

export default function MultiCanvas() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = 400;
    canvas.height = 150;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 16;  // Slightly thicker for better recognition
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }, []);

  const getPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    if (e.touches) {
      const t = e.touches[0];
      return { x: t.clientX - rect.left, y: t.clientY - rect.top };
    }
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const start = (e) => {
    e.preventDefault();
    setIsDrawing(true);
    const ctx = canvasRef.current.getContext('2d');
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  };

  const move = (e) => {
    if (!isDrawing) return;
    e.preventDefault();
    const ctx = canvasRef.current.getContext('2d');
    const pos = getPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  };

  const end = (e) => {
    e.preventDefault();
    setIsDrawing(false);
  };

  const clear = () => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    setPrediction(null);
  };
  const predict = async () => {
    setLoading(true);
    setPrediction(null);
    const dataUrl = canvasRef.current.toDataURL('image/png');
    const apiUrl = process.env.REACT_APP_ML_API_URL || 'https://dk376907-digit-prediction-api.hf.space';
    try {
      const res = await fetch(`${apiUrl}/multipredict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      });
      const data = await res.json();
      if (data.success) {
        setPrediction(data.prediction);
      } else {
        alert('Prediction failed: ' + (data.error || 'unknown error'));
      }
    } catch (err) {
      alert('Error calling API: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Multi-digit Recognizer</h2>
      <canvas
        ref={canvasRef}
        style={{ border: '1px solid #ccc', touchAction: 'none' }}
        onMouseDown={start}
        onMouseMove={move}
        onMouseUp={end}
        onMouseLeave={end}
        onTouchStart={start}
        onTouchMove={move}
        onTouchEnd={end}
      />
      <div style={{ marginTop: 10 }}>
        <button onClick={predict} disabled={loading}>{loading ? 'Predicting...' : 'Predict'}</button>
        <button onClick={clear}>Clear</button>
      </div>
      {prediction && (
        <div style={{ marginTop: 12 }}>
          <h3>Prediction: {prediction}</h3>
        </div>
      )}
    </div>
  );
}
