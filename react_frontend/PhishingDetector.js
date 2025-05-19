import React, { useState } from 'react';

function PhishingDetector() {
  const [email, setEmail] = useState('');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult('');
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email })
    });
    const data = await response.json();
    setResult(data.prediction ? `Prediction: ${data.prediction}` : `Error: ${data.error}`);
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 500, margin: '60px auto', background: '#fff', padding: 30, borderRadius: 8, boxShadow: '0 2px 8px #ccc' }}>
      <h2>Phishing Email Detector</h2>
      <form onSubmit={handleSubmit}>
        <label>Enter Email Text:</label><br />
        <textarea value={email} onChange={e => setEmail(e.target.value)} required style={{ width: '100%', height: 120, marginBottom: 15 }} />
        <button type="submit" style={{ width: '100%', padding: 10, background: '#007bff', color: '#fff', border: 'none', borderRadius: 4, fontSize: 16 }}>Analyze Email</button>
      </form>
      <div style={{ marginTop: 20, textAlign: 'center', fontSize: 18 }}>
        {loading ? 'Analyzing...' : result}
      </div>
    </div>
  );
}

export default PhishingDetector; 