import React, { useRef, useEffect, useState } from 'react';

export default function Canvas() {
  const canvasRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hue, setHue] = useState(0);

  useEffect(() => {
    // Setup visible canvas
    const canvas = canvasRef.current;
    canvas.width = 400; // Increased resolution for better look
    canvas.height = 400;
    const ctx = canvas.getContext('2d');
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 12; // Adjusted for resolution

    // Clear with transparent or black? Prompt asks for dark/colorful, but design is minimalist white.
    // Let's keep the minimalist white background for the App, but maybe the canvas itself is a "portal".
    // Or, on a white background, colorful strokes look great.
    // Let's fill with white to match the new light theme, but maybe give it a subtle texture or just clear.
    // Actually, high contrast neon looks best on dark. 
    // IF the user wants "silky glossy", usually that implies a dark background. 
    // BUT the previous step set the theme to LIGHT white.
    // Reconciling: A white canvas with colorful "gel" pen strokes also looks silky.
    // Let's stick to the current theme (White bg) but make the strokes vibrant.
    // Wait, the prompt said "classic minimalist" previously, but now "silky glossy" usually implies darkness.
    // Let's try a dark canvas *inside* the light UI? Or just colorful strokes on white.
    // Let's go with Dark Canvas for the drawing area specifically, as it makes colors pop.

    ctx.fillStyle = '#0f172a'; // Dark slate/black for contrast
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Setup hidden canvas (Model expects 28x28 usually, or similar aspect ratio. The backend handles resize.
    // But importantly, model expects White digit on Black background).
    const hiddenCanvas = hiddenCanvasRef.current;
    hiddenCanvas.width = 280;
    hiddenCanvas.height = 280;
    const hCtx = hiddenCanvas.getContext('2d');
    hCtx.fillStyle = 'black';
    hCtx.fillRect(0, 0, hiddenCanvas.width, hiddenCanvas.height);
    hCtx.strokeStyle = 'white';
    hCtx.lineWidth = 20;
    hCtx.lineCap = 'round';
    hCtx.lineJoin = 'round';
  }, []);

  const getPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;

    if (e.touches) {
      const t = e.touches[0];
      return {
        x: (t.clientX - rect.left) * scaleX,
        y: (t.clientY - rect.top) * scaleY
      };
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  };

  const draw = (e) => {
    if (!isDrawing) return;
    e.preventDefault();

    const pos = getPos(e);
    const ctx = canvasRef.current.getContext('2d');
    const hCtx = hiddenCanvasRef.current.getContext('2d'); // Hidden canvas context

    // Update Hue
    setHue((prev) => (prev + 1) % 360);
    const color = `hsl(${hue}, 100%, 50%)`;

    // Draw on Visible Canvas (Silky/Glossy)
    ctx.strokeStyle = color;
    ctx.shadowBlur = 10;
    ctx.shadowColor = color;
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    // Reset shadow for next stroke parts to avoid artifacts if needed, 
    // but here we want continuous glow.
    // Important: beginPath is called in start(), lineTo here.
    // To make it really smooth, we might need continuous beginPath/moveTo/lineTo methodology 
    // but standard stroke is okay.
    // For hue changing *during* a stroke, we technically need to start new paths constantly.

    ctx.beginPath(); // Start new path for new color segment
    ctx.moveTo(pos.x, pos.y);


    // Draw on Hidden Canvas (Model Input)
    // We need to map the pos from visible resolution to hidden resolution
    const hScaleX = hiddenCanvasRef.current.width / canvasRef.current.width;
    const hScaleY = hiddenCanvasRef.current.height / canvasRef.current.height;

    hCtx.lineTo(pos.x * hScaleX, pos.y * hScaleY);
    hCtx.stroke();
    hCtx.beginPath();
    hCtx.moveTo(pos.x * hScaleX, pos.y * hScaleY);
  };

  const start = (e) => {
    e.preventDefault();
    setIsDrawing(true);
    const pos = getPos(e);

    const ctx = canvasRef.current.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);

    const hCtx = hiddenCanvasRef.current.getContext('2d');
    const hScaleX = hiddenCanvasRef.current.width / canvasRef.current.width;
    const hScaleY = hiddenCanvasRef.current.height / canvasRef.current.height;
    hCtx.beginPath();
    hCtx.moveTo(pos.x * hScaleX, pos.y * hScaleY);
  };

  const end = (e) => {
    e.preventDefault();
    setIsDrawing(false);
    // Begin path is not needed here as draw/start handles it
  };

  const clear = () => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    const hCtx = hiddenCanvasRef.current.getContext('2d');
    hCtx.fillStyle = 'black';
    hCtx.fillRect(0, 0, hiddenCanvasRef.current.width, hiddenCanvasRef.current.height);

    setPrediction(null);
    setProbabilities(null);
  };

  const predict = async () => {
    console.log('[Canvas] Predict started');
    setLoading(true);
    setPrediction(null);
    setProbabilities(null);
    // Use HIDDEN canvas for prediction
    const dataUrl = hiddenCanvasRef.current.toDataURL('image/png');
    const apiUrl = process.env.REACT_APP_ML_API_URL || 'https://dk376907-digit-prediction-api.hf.space';

    try {
      const res = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      });

      const data = await res.json();
      if (data.success) {
        setPrediction(data.prediction);
        setProbabilities(data.probabilities);
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
      <div className="workspace">
        <div className="canvas-wrapper">
          <canvas
            ref={canvasRef}
            style={{ width: '100%', maxWidth: '400px', borderRadius: '12px' }} // CSS scaling
            onMouseDown={start}
            onMouseMove={draw}
            onMouseUp={end}
            onMouseLeave={end}
            onTouchStart={start}
            onTouchMove={draw}
            onTouchEnd={end}
          />
          {/* Hidden Canvas for Model */}
          <canvas ref={hiddenCanvasRef} style={{ display: 'none' }} />
        </div>

        <div className="controls-panel">
          <button className="primary" onClick={predict} disabled={loading}>
            {loading ? 'Processing...' : 'Predict Digit'}
          </button>
          <button className="secondary" onClick={clear}>
            Clear Canvas
          </button>

          {prediction !== null && (
            <div className="prediction-result">
              <h3>Prediction: {prediction}</h3>
              <div className="probabilities">
                {probabilities && probabilities.map((p, idx) => (
                  <div key={idx} className={`prob-item ${p > 0.5 ? 'high' : ''}`}>
                    {idx}: {(p * 100).toFixed(1)}%
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
