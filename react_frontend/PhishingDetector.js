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
    <div style={{ maxWidth: 600, margin: '60px auto', background: 'rgba(17, 24, 39, 0.7)', padding: 30, borderRadius: 16, boxShadow: '0 8px 24px rgba(0,0,0,0.35)', color: '#e5e7eb', border: '1px solid rgba(148, 163, 184, 0.15)', backdropFilter: 'blur(10px)' }}>
      <h2 style={{
        textAlign: 'center',
        marginBottom: 20,
        fontWeight: 700,
        backgroundImage: 'linear-gradient(45deg, #7c3aed, #ec4899)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent'
      }}>Phishing Email Detector</h2>
      <form onSubmit={handleSubmit}>
        <label style={{ color: '#e5e7eb', fontWeight: 600 }}>Enter Email Text:</label><br />
        <textarea value={email} onChange={e => setEmail(e.target.value)} required style={{ width: '100%', height: 140, marginBottom: 16, padding: 14, background: 'rgba(2, 6, 23, 0.5)', color: '#e5e7eb', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: 10 }} />
        <button type="submit" style={{ width: '100%', padding: 14, backgroundImage: 'linear-gradient(90deg, #7c3aed, #ec4899)', color: '#fff', border: 'none', borderRadius: 10, fontSize: 18, fontWeight: 700, boxShadow: '0 10px 30px rgba(124, 58, 237, 0.25)', cursor: 'pointer' }}>Analyze Email</button>
      </form>
      <div style={{ marginTop: 20, textAlign: 'center', fontSize: 18, color: '#e5e7eb' }}>
        {loading ? 'Analyzing...' : result}
      </div>
    </div>
  );
}

export default PhishingDetector; 