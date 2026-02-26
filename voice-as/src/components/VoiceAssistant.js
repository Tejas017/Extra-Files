// src/components/VoiceAssistant.js
import React, { useState, useEffect, useRef } from 'react';

const VoiceAssistant = ({ listening }) => {
  const [activeArea, setActiveArea] = useState(null);
  const [text1, setText1] = useState('');
  const [text2, setText2] = useState('');

  const recognitionRef = useRef(null);

  useEffect(() => {
    if (!('webkitSpeechRecognition' in window)) {
      alert("Web Speech API not supported in this browser.");
      return;
    }

    const recognition = new window.webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
      const last = event.results.length - 1;
      const transcript = event.results[last][0].transcript.trim().toLowerCase();
      console.log('Heard:', transcript);

      if (transcript.includes("text area one")) {
        setActiveArea("text1");
      } else if (transcript.includes("text area two")) {
        setActiveArea("text2");
      } else if (transcript.includes("clear")) {
        setText1('');
        setText2('');
        setActiveArea(null);
      } else if (transcript.includes("stop")) {
        setActiveArea(null);
      } else {
        if (activeArea === "text1") {
          setText1(prev => prev + " " + transcript);
        } else if (activeArea === "text2") {
          setText2(prev => prev + " " + transcript);
        }
      }
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
    };

    recognitionRef.current = recognition;

    // Inside VoiceAssistant component
    const handleSave = async () => {
    const response = await fetch("http://localhost:5000/save", {
        method: "POST",
        headers: {
        "Content-Type": "application/json",
        },
        body: JSON.stringify({ text1, text2, timestamp: new Date().toISOString() }),
    });

    const result = await response.json();
    alert(result.message);
    };

    if (listening) {
      recognition.start();
    } else {
      recognition.stop();
    }

    return () => {
      recognition.stop();
    };
  }, [listening, activeArea]);

  return (
    <div style={{ padding: '1rem' }}>
      <div>
        <label>Text Area 1</label><br />
        <textarea rows={5} cols={50} value={text1} readOnly />
      </div>
      <br />
      <div>
        <label>Text Area 2</label><br />
        <textarea rows={5} cols={50} value={text2} readOnly />
      </div>
      <button onClick={handleSave}>💾 Save Transcription</button>

    </div>
  );
};

export default VoiceAssistant;
