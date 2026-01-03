import React, { useRef, useEffect, useState } from 'react';

export default function MultiCanvas() {
  const canvasRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hue, setHue] = useState(0);

  useEffect(() => {
    // Setup visible canvas
    const canvas = canvasRef.current;
    canvas.width = 800; // Enlarged width
    canvas.height = 300; // Enlarged height
    const ctx = canvas.getContext('2d');
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 12;

    ctx.fillStyle = '#0f172a'; // Dark background for neon effect
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Setup hidden canvas (Keep ratio, high contrast)
    const hiddenCanvas = hiddenCanvasRef.current;
    hiddenCanvas.width = 800; // Match visible resolution for 1:1 mapping
    hiddenCanvas.height = 300;
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
    const hCtx = hiddenCanvasRef.current.getContext('2d');

    // Update Hue
    setHue((prev) => (prev + 1) % 360);
    const color = `hsl(${hue}, 100%, 50%)`;

    // Visible Canvas
    ctx.strokeStyle = color;
    ctx.shadowBlur = 10;
    ctx.shadowColor = color;
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    // Continuous path for smooth glow
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);

    // Hidden Canvas
    hCtx.lineTo(pos.x, pos.y);
    hCtx.stroke();
    hCtx.beginPath();
    hCtx.moveTo(pos.x, pos.y);
  };

  const start = (e) => {
    e.preventDefault();
    setIsDrawing(true);
    const pos = getPos(e);

    const ctx = canvasRef.current.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);

    const hCtx = hiddenCanvasRef.current.getContext('2d');
    hCtx.beginPath();
    hCtx.moveTo(pos.x, pos.y);
  };

  const end = (e) => {
    e.preventDefault();
    setIsDrawing(false);
  };

  const clear = () => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    const hCtx = hiddenCanvasRef.current.getContext('2d');
    hCtx.fillStyle = 'black';
    hCtx.fillRect(0, 0, hiddenCanvasRef.current.width, hiddenCanvasRef.current.height);

    setPrediction(null);
  };

  const predict = async () => {
    setLoading(true);
    setPrediction(null);
    const dataUrl = hiddenCanvasRef.current.toDataURL('image/png');
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
      <div className="workspace">
        <div className="canvas-wrapper">
          <canvas
            ref={canvasRef}
            style={{ width: '100%', maxWidth: '800px', borderRadius: '12px' }} // CSS scaling
            onMouseDown={start}
            onMouseMove={draw}
            onMouseUp={end}
            onMouseLeave={end}
            onTouchStart={start}
            onTouchMove={draw}
            onTouchEnd={end}
          />
          {/* Hidden Canvas */}
          <canvas ref={hiddenCanvasRef} style={{ display: 'none' }} />
        </div>

        <div className="controls-panel">
          <button className="primary" onClick={predict} disabled={loading}>
            {loading ? 'Processing...' : 'Read Sequence'}
          </button>
          <button className="secondary" onClick={clear}>
            Clear Canvas
          </button>

          {prediction && (
            <div className="prediction-result">
              <h3>Prediction: {prediction}</h3>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
